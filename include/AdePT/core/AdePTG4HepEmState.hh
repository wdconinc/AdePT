// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_G4_HEPEM_STATE_HH
#define ADEPT_G4_HEPEM_STATE_HH

#include <memory>

struct G4HepEmConfig;
struct G4HepEmData;
struct G4HepEmParameters;

namespace AsyncAdePT {

/// @brief Holds the host-side G4HepEm inputs required by AdePT transport.
/// @details
/// The Geant4 integration side prepares one of these objects before the shared
/// transport is created. This wrapper holds:
/// - a borrowed (non-owning) pointer to the `G4HepEmData` from the master
///   `G4HepEmRunManager`. The master already built these tables during Geant4
///   physics-list initialization; AdePT reuses them to avoid repeating the
///   full cross-section table construction (which can take >10 minutes for
///   complex geometries). The master RunManager frees the data at Geant4 shutdown.
/// - an owned deep copy of the `G4HepEmParameters` taken from the provided config
///
/// Cleanup is intentionally split:
/// - `DataDeleter` is a no-op (borrowed pointer — master RunManager owns the data).
/// - `ParametersDeleter` performs the deep cleanup of the owned
///   `G4HepEmParameters`, including the GPU mirror created by AdePT, and then
///   deletes the outer `G4HepEmParameters` allocation.
class AdePTG4HepEmState {
public:
  /// @brief Prepare AdePT's G4HepEm inputs by borrowing from the master `G4HepEmRunManager`
  ///        and deep-copying the `G4HepEmParameters` from the supplied config.
  explicit AdePTG4HepEmState(G4HepEmConfig *hepEmConfig);

  /// @brief Destroy the borrowed `G4HepEmData` reference (no-op) and the owned `G4HepEmParameters` copy.
  ~AdePTG4HepEmState();

  AdePTG4HepEmState(const AdePTG4HepEmState &)            = delete;
  AdePTG4HepEmState &operator=(const AdePTG4HepEmState &) = delete;

  AdePTG4HepEmState(AdePTG4HepEmState &&) noexcept            = default;
  AdePTG4HepEmState &operator=(AdePTG4HepEmState &&) noexcept = default;

  /// @brief Access the owned host-side HepEm data tables.
  G4HepEmData *GetData() const { return fData.get(); }

  /// @brief Access the owned HepEm parameter copy.
  G4HepEmParameters *GetParameters() const { return fParameters.get(); }

private:
  /// @brief Optionally deletes the `G4HepEmData` object after first freeing all tables it owns.
  /// @details
  /// `FreeG4HepEmData` releases both the host-side tables and any device-side
  /// mirrors embedded in the `G4HepEmData` object, but it does not delete the outer
  /// `G4HepEmData` allocation itself. This deleter performs both steps when `owned`
  /// is true (the data was rebuilt by AdePT). When `owned` is false the pointer is
  /// borrowed from the master `G4HepEmRunManager` and must not be freed here.
  struct DataDeleter {
    bool owned = true;
    void operator()(G4HepEmData *data) const;
  };

  /// @brief Deletes the outer `G4HepEmParameters` object after first freeing
  /// all host/device allocations it owns.
  /// @details
  /// The copied parameter block owns its `fParametersPerRegion` host array and
  /// the GPU mirror pointed to by `fParametersPerRegion_gpu` after transport
  /// upload. `FreeG4HepEmParameters` releases those nested allocations, while
  /// this deleter also deletes the outer `G4HepEmParameters` allocation.
  struct ParametersDeleter {
    void operator()(G4HepEmParameters *parameters) const;
  };

  /// Non-owning borrowed pointer to the master `G4HepEmRunManager`'s `G4HepEmData`.
  /// The master RunManager built the tables during Geant4 physics-list initialization
  /// and is responsible for freeing them at shutdown. The `DataDeleter` is a no-op.
  std::unique_ptr<G4HepEmData, DataDeleter> fData;

  /// Owned deep copy of `G4HepEmParameters` used to build and upload transport data.
  std::unique_ptr<G4HepEmParameters, ParametersDeleter> fParameters;
};

} // namespace AsyncAdePT

#endif
