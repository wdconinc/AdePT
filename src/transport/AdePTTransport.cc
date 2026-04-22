// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/AdePTTransport.hh>
#include <AdePT/transport/queues/TrackBuffer.hh>

#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/UnplacedBox.h>
#ifdef ADEPT_USE_SURF
#include <VecGeom/surfaces/BrepHelper.h>
#endif
#ifdef ADEPT_ENABLE_NSYS_PROFILING
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace adept::transport::detail {
void setDeviceLimits(int stackLimit = 0, int heapLimit = 0);
void CopySurfaceModelToGPU();
void InitWDTOnDevice(const adeptint::WDTHostPacked &, adeptint::WDTDeviceBuffers &, unsigned short);
void UploadG4HepEmToGPU(G4HepEmData *hepEmData, G4HepEmParameters *hepEmParameters);
std::thread LaunchGPUWorker(int, int, int, adept::transport::TrackBuffer &, adept::transport::GPUstate &,
                            std::vector<std::atomic<adept::transport::EventState>> &, std::condition_variable &, int,
                            int, bool, bool, unsigned short, const double, bool);
std::unique_ptr<adept::transport::GPUstate, adept::transport::GPUstateDeleter> InitializeGPU(
    int trackCapacity, int stepCapacity, int numThreads, adept::transport::TrackBuffer &trackBuffer,
    double CPUCapacityFactor, double CPUCopyFraction, std::string &generalBfieldFile,
    const std::vector<float> &uniformBfieldValues);
void FreeGPU(std::unique_ptr<adept::transport::GPUstate, adept::transport::GPUstateDeleter> &, std::thread &,
             adeptint::WDTDeviceBuffers &);

#ifdef ADEPT_ENABLE_NSYS_PROFILING
// AdePT owns one active transport loop at a time, so profiler capture state is
// intentionally process-global and unsynchronized. If multiple concurrent
// transport loops are introduced, protect this state with an atomic or mutex.
static bool gTransportProfilerStarted = false;

bool StartTransportProfilerCapture()
{
  if (gTransportProfilerStarted) return false;
  const auto result = cudaProfilerStart();
  if (result != cudaSuccess) {
    std::cerr << "AdePTTransport: cudaProfilerStart failed: " << cudaGetErrorString(result)
              << "; disabling transport profiler capture" << std::endl;
    return false;
  }
  gTransportProfilerStarted = true;
  std::cout << "AdePT: CUDA profiler capture started" << std::endl;
  return true;
}

void StopTransportProfilerCapture()
{
  if (!gTransportProfilerStarted) return;
  const auto syncResult = cudaDeviceSynchronize();
  if (syncResult != cudaSuccess) {
    std::cerr << "AdePTTransport: cudaDeviceSynchronize before cudaProfilerStop failed: "
              << cudaGetErrorString(syncResult) << std::endl;
  }
  const auto result = cudaProfilerStop();
  if (result != cudaSuccess) {
    std::cerr << "AdePTTransport: cudaProfilerStop failed: " << cudaGetErrorString(result) << std::endl;
  }
  gTransportProfilerStarted = false;
}
#endif
} // namespace adept::transport::detail

namespace adept::transport {

namespace {

constexpr unsigned int kMaxSupportedTransportThreads = 10000u;

unsigned int ValidateNumThreads(unsigned int numThreads)
{
  if (numThreads == 0) throw std::invalid_argument("AdePTTransport requires a positive number of threads");
  if (numThreads > kMaxSupportedTransportThreads) {
    throw std::invalid_argument("AdePTTransport supports up to " + std::to_string(kMaxSupportedTransportThreads) +
                                " threads");
  }
  return numThreads;
}

} // namespace

AdePTTransport::AdePTTransport(const AdePTTransportConfig &configuration,
                               std::unique_ptr<AdePTG4HepEmState> adeptG4HepEmState, adeptint::VolAuxData *auxData,
                               const adeptint::WDTHostPacked &wdtPacked, const std::vector<float> &uniformFieldValues)
    : fAdePTSeed{configuration.adeptSeed}, fNThread{ValidateNumThreads(configuration.numThreads)},
      fTrackCapacity{configuration.trackCapacity}, fStepCapacity{configuration.stepCapacity},
      fDebugLevel{configuration.debugLevel}, fCUDAStackLimit{configuration.cudaStackLimit},
      fCUDAHeapLimit{configuration.cudaHeapLimit}, fLastNParticlesOnCPU{configuration.lastNParticlesOnCPU},
      fMaxWDTIter{configuration.maxWDTIter}, fAdePTG4HepEmState(std::move(adeptG4HepEmState)), fEventStates(fNThread),
      fReturnAllSteps{configuration.returnAllSteps}, fReturnFirstAndLastStep{configuration.returnFirstAndLastStep},
      fBfieldFile{configuration.bfieldFile}, fCPUCapacityFactor{configuration.cpuCapacityFactor},
      fCPUCopyFraction{configuration.cpuCopyFraction}, fStepBufferSafetyFactor{configuration.stepBufferSafetyFactor}
{
  for (auto &eventState : fEventStates) {
    std::atomic_init(&eventState, EventState::DeviceFlushed);
  }

  Initialize(auxData, wdtPacked, uniformFieldValues);
}

AdePTTransport::~AdePTTransport()
{
  adept::transport::detail::FreeGPU(std::ref(fGPUstate), fGPUWorker, fWDTDev);
}

void AdePTTransport::AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y, double z,
                              double dirx, double diry, double dirz, double globalTime, double localTime,
                              double properTime, float weight, unsigned short stepCounter, int threadId,
                              unsigned int eventId, vecgeom::NavigationState &&state)
{
  if (pdg != 11 && pdg != -11 && pdg != 22) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": Only supporting EM tracks. Got pdgID=" << pdg << "\n";
    return;
  }

  adeptint::TrackData track{pdg,         trackId,
                            parentId,    energy,
                            x,           y,
                            z,           dirx,
                            diry,        dirz,
                            globalTime,  localTime,
                            properTime,  weight,
                            stepCounter, std::move(state),
                            eventId,     static_cast<short>(threadId)};

  {
    auto trackHandle  = fBuffer->createToDeviceSlot();
    trackHandle.track = std::move(track);
  }

  fEventStates[threadId].store(EventState::NewTracksFromG4, std::memory_order_release);
}

bool AdePTTransport::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world, const adeptint::VolAuxData *auxData,
                                        size_t numVolumes)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  adept::transport::detail::setDeviceLimits(fCUDAStackLimit, fCUDAHeapLimit);

  bool success = true;
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_MIXED_PRECISION
  using SurfData   = vgbrep::SurfData<float>;
  using BrepHelper = vgbrep::BrepHelper<float>;
#else
  using SurfData   = vgbrep::SurfData<double>;
  using BrepHelper = vgbrep::BrepHelper<double>;
#endif
  auto start = std::chrono::steady_clock::now();
  if (!BrepHelper::Instance().Convert()) return false;
  BrepHelper::Instance().PrintSurfData();
  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  std::cout << "== Conversion to surface model done in " << elapsed.count() << " [s]\n";
  cudaManager.SynchronizeNavigationTable();
  adept::transport::detail::CopySurfaceModelToGPU();
#else
  // Before copying to GPU, scan all logical volumes and replace any shapes
  // that cannot be serialized to GPU (DeviceSizeOf() == 0) with conservative
  // bounding-box approximations.  The decision is region-based:
  //   - Volume in a GPU region with an incompatible shape: fatal error.
  //   - Volume outside all GPU regions with an incompatible shape: safe to
  //     approximate because GPU tracks never navigate inside those volumes.
  // When fTrackInAllRegions is true the auxData region flags are not used and
  // every volume must be GPU-serializable, so no substitution is attempted.
  std::vector<std::pair<vecgeom::LogicalVolume *, vecgeom::VUnplacedVolume const *>> replacedVolumes;
  if (!fTrackInAllRegions && auxData != nullptr) {
    std::vector<vecgeom::LogicalVolume *> allLV;
    vecgeom::GeoManager::Instance().GetAllLogicalVolumes(allLV);
    for (auto *lv : allLV) {
      const auto *uv = lv->GetUnplacedVolume();
      if (uv->DeviceSizeOf() == 0) {
        const unsigned int id  = lv->id();
        const bool inGPURegion = (id < numVolumes) && (auxData[id].fGPUregionId >= 0);
        if (inGPURegion) {
          throw std::runtime_error(std::string("AdePTTransport::InitializeGeometry: volume ") +
                                   lv->GetName() + " is inside a GPU region but its shape cannot be "
                                   "serialized to the GPU. This configuration is not supported.");
        }
        vecgeom::Vector3D<vecgeom::Precision> aMin, aMax;
        uv->Extent(aMin, aMax);
        // Use max(|aMin|, |aMax|) per axis: the box is centered at origin but
        // tessellated meshes may not be, so we inflate to cover the full mesh.
        constexpr vecgeom::Precision kMinHalf = 1.0; // 1 mm in VecGeom units
        const vecgeom::Precision hx = std::max({std::abs(aMin.x()), std::abs(aMax.x()), kMinHalf});
        const vecgeom::Precision hy = std::max({std::abs(aMin.y()), std::abs(aMax.y()), kMinHalf});
        const vecgeom::Precision hz = std::max({std::abs(aMin.z()), std::abs(aMax.z()), kMinHalf});
        auto *bbox = new vecgeom::UnplacedBox(hx, hy, hz);
        std::cout << "AdePT: Replacing volume \"" << lv->GetName() << "\" (id=" << lv->id()
                  << ") with bbox [" << hx << "," << hy << "," << hz << "] mm\n";
        replacedVolumes.push_back({lv, lv->SetUnplacedVolume(bbox)});
      }
    }
    if (!replacedVolumes.empty())
      std::cout << "AdePT: Replaced " << replacedVolumes.size()
                << " volume(s) outside GPU regions with bounding boxes for GPU geometry copy\n";
  }

  cudaManager.LoadGeometry(world);
  auto world_dev = cudaManager.Synchronize();
  success        = world_dev != nullptr;
  InitBVH();

  // Restore original CPU-side unplaced volumes; the GPU retains the bounding-box copies.
  for (auto &[lv, original] : replacedVolumes) {
    delete lv->SetUnplacedVolume(original);
  }
#endif
  return success;
}

bool AdePTTransport::InitializePhysics()
{
  if (!fAdePTG4HepEmState) {
    throw std::runtime_error("AdePTTransport::InitializePhysics: Missing AdePT-owned G4HepEm state.");
  }

  adept::transport::detail::UploadG4HepEmToGPU(fAdePTG4HepEmState->GetData(), fAdePTG4HepEmState->GetParameters());
  return true;
}

void AdePTTransport::Initialize(adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                                const std::vector<float> &uniformFieldValues)
{
  if (vecgeom::GeoManager::Instance().GetRegisteredVolumesCount() == 0)
    throw std::runtime_error("AdePTTransport::Initialize: Number of geometry volumes is zero.");

  std::cout << "=== AdePTTransport: initializing geometry and physics\n";
  if (!vecgeom::GeoManager::Instance().IsClosed())
    throw std::runtime_error("AdePTTransport::Initialize: VecGeom geometry not closed.");

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!InitializeGeometry(world, auxData, vecgeom::GeoManager::Instance().GetRegisteredVolumesCount()))
    throw std::runtime_error("AdePTTransport::Initialize: Cannot initialize geometry on GPU");

  if (!InitializePhysics()) throw std::runtime_error("AdePTTransport::Initialize cannot initialize physics on GPU");

  const auto numVolumes   = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  auto &volAuxArray       = adeptint::VolAuxArray::GetInstance();
  volAuxArray.fNumVolumes = numVolumes;
  volAuxArray.fAuxData    = auxData;
  adept::transport::InitVolAuxArray(volAuxArray);

  fHasWDTRegions = !wdtPacked.regions.empty();

  adeptint::WDTDeviceBuffers wdtDev;
  adept::transport::detail::InitWDTOnDevice(wdtPacked, wdtDev, fMaxWDTIter);
  fWDTDev = wdtDev;

  const auto toDeviceSlots = 4u * 8192u * fNThread;
  std::cout << "\nAllocating " << toDeviceSlots << " To-device buffer slots\n";
  fBuffer = std::make_unique<TrackBuffer>(toDeviceSlots);

  assert(fBuffer != nullptr);

  fGPUstate =
      adept::transport::detail::InitializeGPU(fTrackCapacity, fStepCapacity, fNThread, *fBuffer, fCPUCapacityFactor,
                                              fCPUCopyFraction, fBfieldFile, uniformFieldValues);
  fGPUWorker = adept::transport::detail::LaunchGPUWorker(fTrackCapacity, fStepCapacity, fNThread, *fBuffer, *fGPUstate,
                                                         fEventStates, fCV_G4Workers, fAdePTSeed, fDebugLevel,
                                                         fReturnAllSteps, fReturnFirstAndLastStep, fLastNParticlesOnCPU,
                                                         fStepBufferSafetyFactor, fHasWDTRegions);
}

void AdePTTransport::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdePTTransport::RequestFlush(int threadId)
{
  fEventStates[threadId].store(EventState::G4RequestsFlush, std::memory_order_release);
}

void AdePTTransport::WaitForFlushProgress()
{
  std::unique_lock lock{fMutex_G4Workers};
  using namespace std::chrono_literals;
  fCV_G4Workers.wait_for(lock, 1ms);
}

bool AdePTTransport::AreReturnedStepsFlushed(int threadId) const
{
  return fEventStates[threadId].load(std::memory_order_acquire) >= EventState::StepsFlushed;
}

void AdePTTransport::MarkHostFlushed(int threadId)
{
  fEventStates[threadId].store(EventState::DeviceFlushed, std::memory_order_release);
}

} // namespace adept::transport
