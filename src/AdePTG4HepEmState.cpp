// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTG4HepEmState.hh>

#include <G4HepEmConfig.hh>
#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRunManager.hh>
#include <G4ios.hh>

#include <algorithm>
#include <stdexcept>

namespace AsyncAdePT {

/// @brief Release the tables owned by `G4HepEmData` and then delete the outer object.
/// @details
/// This is the cleanup path for the fully owned `fData` member. It is separate
/// from the class destructor because `std::unique_ptr` needs a deleter for the
/// deep cleanup before the outer `G4HepEmData` allocation itself can be deleted.
void AdePTG4HepEmState::DataDeleter::operator()(G4HepEmData *data) const
{
  if (!owned || data == nullptr) return;
  FreeG4HepEmData(data);
  delete data;
}

/// @brief Release the copied HepEm parameters and then delete the outer object.
/// @details
/// The copied `G4HepEmParameters` block is fully owned by
/// `AdePTG4HepEmState`. This deep
/// cleanup therefore releases both the host-side per-region array and the
/// device-side mirror created during transport upload before deleting the outer
/// `G4HepEmParameters` allocation itself.
void AdePTG4HepEmState::ParametersDeleter::operator()(G4HepEmParameters *parameters) const
{
  if (parameters == nullptr) return;
  FreeG4HepEmParameters(parameters);
  delete parameters;
}

/// @brief Prepare the AdePT host-side G4HepEm inputs by borrowing from the already-initialized
///        master G4HepEmRunManager and deep-copying the G4HepEmParameters from the supplied config.
/// @details
/// `AdePTG4HepEmState` holds two G4HepEm objects:
/// - a borrowed (non-owning) pointer to the `G4HepEmData` owned by the master
///   `G4HepEmRunManager`. The master built these tables during Geant4 physics-list
///   initialization; rebuilding them here would duplicate the entire cross-section
///   table construction (which can exceed 10 minutes for complex geometries such as
///   ePIC no_bhcal). The master RunManager is responsible for freeing this data at
///   Geant4 shutdown, after AdePT transport has already been destroyed.
/// - a deep copy of the `G4HepEmParameters` stored in the supplied `G4HepEmConfig`.
///   We must copy the parameters because the original object remains owned by the
///   worker-local `G4HepEmConfig`, while the shared AdePT transport can outlive
///   the worker that first created it.
AdePTG4HepEmState::AdePTG4HepEmState(G4HepEmConfig *hepEmConfig)
    : fData(nullptr, DataDeleter{false}), fParameters(new G4HepEmParameters)
{
  if (hepEmConfig == nullptr) {
    throw std::runtime_error("AdePTG4HepEmState requires a non-null G4HepEmConfig.");
  }

  G4HepEmParameters *sourceParameters = hepEmConfig->GetG4HepEmParameters();
  if (sourceParameters == nullptr) {
    throw std::runtime_error("AdePTG4HepEmState requires initialized G4HepEmParameters in the supplied config.");
  }

  // Deep-copy the G4HepEmParameters so the shared transport does not keep a
  // pointer into a worker-owned G4HepEmConfig.
  *fParameters                      = *sourceParameters;
  fParameters->fParametersPerRegion = nullptr;
#ifdef G4HepEm_CUDA_BUILD
  fParameters->fParametersPerRegion_gpu = nullptr;
#endif
  if (sourceParameters->fNumRegions > 0) {
    if (sourceParameters->fParametersPerRegion == nullptr) {
      throw std::runtime_error("AdePTG4HepEmState requires initialized per-region G4HepEmParameters.");
    }
    fParameters->fParametersPerRegion = new G4HepEmRegionParmeters[sourceParameters->fNumRegions];
    std::copy_n(sourceParameters->fParametersPerRegion, sourceParameters->fNumRegions,
                fParameters->fParametersPerRegion);
  }

  // Borrow the already-initialized G4HepEmData from the master G4HepEmRunManager
  // instead of rebuilding it. Rebuilding would duplicate the full cross-section
  // table construction that Geant4's physics-list initialization already performed.
  G4HepEmRunManager *masterRM = G4HepEmRunManager::GetMasterRunManager();
  if (masterRM == nullptr || masterRM->GetHepEmData() == nullptr) {
    throw std::runtime_error("AdePTG4HepEmState: master G4HepEmRunManager has no initialized G4HepEmData.");
  }
  fData.reset(masterRM->GetHepEmData());

  G4HepEmMatCutData *cutData = fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;
}

AdePTG4HepEmState::~AdePTG4HepEmState() = default;

} // namespace AsyncAdePT
