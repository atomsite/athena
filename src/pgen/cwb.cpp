//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cwb.cpp
//! \brief Problem generator for 3D Colliding Wind Binary problem with dust advection and
//!        growth, simulation is 3D cartesian only, and simulates Keplerian orbits
//!        

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>    // strcmp()
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

// Preprocessor definitions
// Simulation constants
#define NWIND 2 // Number of winds in simulation
// Indices for scalars, makes for more readable scalar
#define CLOC 0  // Index for wind "colour"
#define ZLOC 1  // Index for dust-to-gas mass ratio, z
#define ALOC 2  // Index for dust grain radius, a
#define LLOC 3  // Index for level scalar
// Indices for stars
#define WR 0 // Index for Wolf-Rayet, always primary star
#define OB 1 // Index for OB star, always secondary
// Conversions
#define MSOLTOGRAM 1.9884099e+33  // Solar mass to gram 
#define MSOLYRTOGS 6.3010252e+25  // Solar mass per year to grams per second
#define MICRONTOCM 1.0000000e-04  // Micron to cm
#define CMTOMICRON 1.0000000e+04  // cm to micron (used as much as micron to cm, divide slower)
// Physical Constants
// Pi is already defined in defs.hpp!
#define KBOLTZ 1.3806490e-16 // Boltmann constant in CGS (erg/K)
#define MASSH  1.6735575e-24 // Hydrogen mass (g)
#define MASSE  9.1093836e-28 // Electron mass (g)
#define G      6.6743000e-8  // Gravitational constant in CGS (dyn cm^2/g^2)
// Functions
#define CUBE(x) ( (x)*(x)*(x) )  // Preprocessor cube function, faster than pow(x,3.0)
// End of preprocessor definitions

// Classes

//! \class Star
//! \brief A class used to contain star and associated wind parameters for a star
//! Both physical properties of the star considered by the simulation (mass, position,
//! velocity etc.) and wind properties (mass loss rate, wind terminal velocity, wind 
//! abundances) are stored here. Some parameters are defined in the problem file in 
//! natural units such as solarmass/year, these are converted to CGS in order to fit
//! more easily into the physics, this must be done manually in ProblemGenerator()
class Star {
  public:
    Real mass; // Star mas (g) 
    Real mdot; // Mass loss rate (g/s)
    Real vinf; // Wind terminal velocity (cm/s)
    Real mom;  // Wind momentum (g cm / s^2)
    Real Twind  = 1e4;            // Wind temperature (K)
    Real pos[3] = {0.0,0.0,0.0};  // X Y and Z coordinates for star (Cartesian, cm/s)
    Real vel[3] = {0.0,0.0,0.0};  // X Y and Z velocity components of star motion
    Real col = 0.0; // Wind colour, 1 for primary star 0 for secondary
    // Wind abundances, stored in an array with the following indexes:
    // 0: Hydrogen
    // 1: Helium
    // 2: Carbon
    // 3: Nitrogen
    // 4: Oxygen
    // Without modification, assumes a pure hydrogen flow
    // Assume that all values should come to around 1.0, all other elements are trace
    Real mass_frac[5] = {1.0,0.0,0.0,0.0,0.0}; // Mass fraction
    Real norm_n_E[5]  = {0.0,0.0,0.0,0.0,0.0}; // Normalised number density
    Real avg_mass     = MASSH;  // Average particle mass (g)
    Real mu           = 1.0;    // Mean molecular mass
    int  ncells_remap = 10;     // Number of cells width for remap radius
    // Methods
    // Calculate average particle mass (g) and mean molecular mass, mu (n*MASSH)
    void calcAvgMass(void) {
      // Calculate masses of each element being considered
      const Real m_E[5] = {1.00*MASSH,  // Hydrogen mass
                           4.00*MASSH,  // Helium mass
                           12.0*MASSH,  // Carbon mass
                           14.0*MASSH,  // Nitrogen mass 
                           16.0*MASSH}; // Oxygen mass
      // Begin processing, 
      Real avgm = 0.0;
      for (int n = 0; n < 5; n++) {
        avgm        += m_E[n] * mass_frac[n];
        norm_n_E[n]  = mass_frac[n] / m_E[n];
      }
      avg_mass = avgm;
      mu       = avgm/MASSH;
    }
    // Short method to calculate wind momentum, needs to be converted to CGS first
    void calcWindMomentum(void) {
      mom = mdot * vinf;
    }
};

//! \class DustDefaults
//! \brief Class used to contain typical minimum and problem-specific initial values
//! Typical minimum values are used to constrain the dust production to realistic values,
//! (grains cannot be subatomic etc etc.)
class DustDefaults {
  public:
    bool  enabled = false;
    // Constants
    const Real z_min  = 1e-8; // Minimum dust-to-gas mass ratio
    const Real a_min  = 1e-6; // Minimum grain radius (micron)
    const Real rho_Gr = 3.0;  // Grain bulk density (g/cm^3)
    const Real A      = 12.0; // Carbon atom mass, AMU
    const Real eps_a  = 0.1;  // Grain sticking probability
    // Variables
    Real z_init = z_min; // Initial d2g mass ratio, defined in problem file
    Real a_init = a_min; // Initial grain radius, defined in problem file
};

//! \class WindCollisionRegion
//! \brief Class for the storage of values relevant to the WCR
//! Class is used to store position of stagnation point, separation distances
//! for stars, separation from stars to stagnation point etc.
//! Positions are updated at the end of every orbitCalc() call
class WindCollisionRegion {
  public:
    // Variables
    Real d_sep;     // Separation distance (cm)
    Real frac_rwr;  // Fraction from stagnation point to WR star
    Real frac_rob;  // Fraction from stagnation point to OB star
    Real r_wr;      // Distance from stagnation point to WR star (cm)
    Real r_ob;      // Distance from stagnation point to OB star (cm)
    Real eta;       // Wind momentum ratio
    // Arrays
    Real pos[3]  = {0.0,0.0,0.0}; // Stagnation point cartesian position (cm)
    Real dist[3] = {0.0,0.0,0.0}; // Star separation components (cm)
    // Functions
    void updatePositions(Star star[]) {
      // Calculate components of separation and stagnation point position
      Real sep2 = 0.0;
      for (int i = 0; i < 3; i++) {
        dist[i] = star[WR].pos[i] - star[OB].pos[i];
        pos[i]  = star[OB].pos[i] - frac_rob*(star[OB].pos[i] - star[WR].pos[i]);
        // Summate dist for each dimension to find dsep^2
        sep2 += SQR(dist[i]); 
      }
      // Update dsep and distances
      d_sep = sqrt(sep2);
      r_ob  = frac_rob * d_sep;
      r_wr  = frac_rwr * d_sep;
    }
    void calcFracs(void) {
      frac_rob = 1.0 - 1.0/(1.0 + sqrt(eta));
      frac_rwr = 1.0 - sqrt(eta)/(1.0 + sqrt(eta));
    }
};

//! \class Cooling
//! \brief Class to enable or disable cooling
//! Currently does not do much, probably shouldn't be a class, but the other
//! features have their own class and I didn't want it to feel left out
//! Could contain CoolCurve as a subclass
class Cooling {
  public:
    bool enabled = false;
};

//! \class CoolCurve
//! \brief Class containing a logarithmically evenly spaced cooling curve
//! Lookup table used to quickly calculate energy loss without time consuming
//! emissivities calculation, to improve accuracy a linear interpolation is used,
//! paramaeters for linear interpolation are found using a binary search
//! using the searchAndInterpolate() function.
//! Contains its own method to read in a plasma cooling curve file in the form
//!   log(T)   Lambda
//! Lambda is normalised with respect to a hydrogen flow with a density of 1 g/cm^3
//! to calculate energy loss use the formulae
//! dE(T)/dt = (rho / massH)^2 * Lambda(T)
class CoolCurve {
  public:
    std::string curve_file_name;
    // Cooling curves
    std::vector<Real> logT; // Cooling curve log(T) values (log(K))
    std::vector<Real> T;    // Cooling curve temperature values (K)
    std::vector<Real> L;    // Cooling curve lambda values
    // Other cooling curve parameters
    Real T_min = 0.0;  // Minimum temperature in cooling curve (K)
    Real T_max = 1e9;  // Maximum temperature in cooling curve (K)
    int  n_bins;       // Number of bins in cooling curve
    // Functions
    // Function to read in cooling curve
    void readPlasmaCoolingCurve(void) {
      // Read in temperature bins
      std::ifstream cool_curve(curve_file_name);
      if (!cool_curve) {
        std::cerr << "Failed to open cooling curve -> " << curve_file_name << "\n";
        std::cerr << "Exiting...\n";
        exit(EXIT_FAILURE);
      }
      cool_curve.seekg(0);
      while (!cool_curve.eof()) {
        Real logT_buf;  // log(T) buffer
        Real L_buf;     // Lambda buffer
        cool_curve >> logT_buf >> L_buf;
        // Add to arrays
        logT.push_back(logT_buf);
        L.push_back(L_buf);
      }
      // Close file
      cool_curve.close();
      // Get number of points in cooling curve
      if (L.size() != logT.size()) {
        std::cerr << "!!! Mismatch in cooling curve array sizes!\n";
        std::cerr << "!!! Exiting...\n";
        exit(EXIT_FAILURE);
      }
      n_bins = logT.size();
      // Create temperature array
      for (int n = 0; n < n_bins; n++) {
        T.push_back(pow(10.0,logT[n]));
      }
      // Ensure temperature bins are ascending and monotonic
      if (!std::is_sorted(T.begin(),T.end())) {
        std::cerr << "!!! Cooling curve " << curve_file_name << " failed test to see if it is sorted!\n";
        std::cerr << "!!! Exiting...\n";
        exit(EXIT_FAILURE);
      }
      T_min = T.front();
      T_max = T.back();
      return;
    }
};

//! \class DustCooling
//! \brief Class to enable or disable dust cooling
//! Currently does not do much, probably shouldn't be a class, but the other
//! features have their own class and I didn't want it to feel left out
//! Could contain IonCurve as a subclass
class DustCooling {
  public:
    bool enabled = false;
};

//! \class IonCurve
//! \brief Class containing a logarithmically temperature spaced ionisation fraction curve
//! Class is used to store a temperature dependent ionisation fraction curve
//! Ionisation fraction refined as ratio of free electrons to ions and is used to
//! accurately calculate the number of free electrons in order to calculate the dust grain
//! heating due to impinging electrons
//! Binary search then linear interpolation is used to determine a reasonably accurate
//! value using the searchAndInterpolate() function.
//! Contains a method to read in a data file with the form
//!     log(T)  ne/ni
class IonCurve {
  public:
    std::string curve_file_name; // String for filename of ion frac
    // Ionisation fraction curve
    std::vector<Real> logT;  // Ionisation curve log(T) (log(K))
    std::vector<Real> T;     // Ionisation curve temperature (K)
    std::vector<Real> E;    // Ionisation curve free electrons per ion
    // Dustcooling functions and such can be stored here
    Real T_min = 0.0;
    Real T_max = 0.0;
    Real E_min = 0.0;
    Real E_max = 0.0;
    int  n_bins;
    // Functions
    // Function to read in ionisation curve
    void readIonCurve(void) {
      // Read in temperature bins
      std::ifstream ion_curve(curve_file_name);
      if (!ion_curve) {
        std::cerr << "Failed to open ion curve -> " << curve_file_name << "\n";
        std::cerr << "Exiting...\n";
        exit(EXIT_FAILURE);
      }
      ion_curve.seekg(0);
      while (!ion_curve.eof()) {
        Real logT_buf;  // log(T) buffer
        Real E_buf;     // Lambda buffer
        ion_curve >> logT_buf >> E_buf;
        // Add to arrays
        logT.push_back(logT_buf);
        E.push_back(E_buf);
      }
      // Close file
      ion_curve.close();
      // Get number of points in cooling curve
      if (E.size() != logT.size()) {
        std::cerr << "!!! Mismatch in ion curve array sizes!\n";
        std::cerr << "!!! Exiting...\n";
        exit(EXIT_FAILURE);
      }
      n_bins = logT.size();
      // Create temperature array
      for (int n = 0; n < n_bins; n++) {
        T.push_back(pow(10.0,logT[n]));
      }
      // Ensure temperature bins are ascending and monotonic
      if (!std::is_sorted(T.begin(),T.end())) {
        std::cerr << "!!! Cooling curve " << curve_file_name << " failed test to see if it is sorted!\n";
        std::cerr << "!!! Exiting...\n";
        exit(EXIT_FAILURE);
      }
      T_min = T.front();
      T_max = T.back();
      E_min = E.front();
      E_max = E.back();
      return;
    }
};

//! \class AMR
//! \brief Class detailing parameters for adaptive mesh refinement
//! This class isn't used for much right now, aside from enabling or disabling AMR
//! However more advanced AMR models that are aware of separation could use this
//! in the future.
class AMR {
  public:
    bool enabled                 = false;
    int  min_cells_between_stars = 150;
    int  G0_level                = 0;      // Maximum level
    // Constants
    const int star_separation = 10; // Number of cells distance to star to refine
    const int wcr_separation  = 10; // Number of cells distance to WCR to refine
    Real G0dx;
    Real finest_dx;
    int max_refine_level;
    int max_level_to_refine;
};

// =======================================================================================
// Function declarations
// =======================================================================================

// Meshblock defined functions do not need declaration, but problem generator functions do

int refinementCondition(MeshBlock *pmb);
void physicalSources(MeshBlock *pmb, const Real time, const Real dt,
                     const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &prim_scalar,
                     const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &cons,
                     AthenaArray<Real> &cons_scalar);
void radiateApprox(MeshBlock *pmb, const Real dt,
                   AthenaArray<Real> &cons);
Real calcGrainCoolRate(Real rho_G, Real a, Real T, Star star, IonCurve ion_curve);
void restrictCool(int is,int ie,
                  int js,int je,
                  int ks,int ke,
                  int nd,
                  Real gmma1,
                  AthenaArray<Real> &dei,
                  const AthenaArray<Real> &cons);
void adjustPressureDueToCooling(int is,int ie,
                                int js,int je,
                                int ks,int ke,
                                Real gmma1,
                                AthenaArray<Real> &dei,
                                AthenaArray<Real> &avgm,
                                AthenaArray<Real> &cons);
void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
void orbitCalc(Real t);
Real searchAndInterpolate(std::vector<Real> x_array, std::vector<Real> y_array, Real x);

// =======================================================================================
// Global variables
// =======================================================================================

// Code is designed to use a number of global objects, and as few global variables as
// possible
// Much of objects are held in global, however MPI schema requires memory pools for these

// Stars
Star star[NWIND];  // Two stars, WR for index 0 and OB for index 1
Real period;       // Orbital period (s)
Real ecc;          // Orbit eccentricity
Real phaseoff;     // Phase offset
// Make wind collision region object
WindCollisionRegion wcr;
// Dust properties
DustDefaults dust; // Initial dust properties
// Cooling objects
Cooling cooling;
CoolCurve cool_curve[NWIND];
DustCooling dust_cooling;
IonCurve ion_curve[NWIND];
// Refinement
AMR amr;

// =======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initialise problem, read in problem file and enroll functions
//! At the start of a run or while re-initialising from a restart file, each thread reads
//! in parameters from input file to classes and global variables, A loob would be more
//! Preferable, but this is fairly readible.
//! Variables are also converted from more convenient units to CGS values, where
//! applicable, and additional calculations are performed, such as the first orbit
//! calculation. The refinemenet and source term functions are also enrolled into the
//! MeshBlock:: class.
//! After reading and initialisation of classes, this function then outputs the input
//! parameters to the terminal, at the immediate end of this function processing begins
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read input file
  // WR wind variables
  star[WR].mass = pin->GetReal("problem","mstar1");
  star[WR].mdot = pin->GetReal("problem","mdot1");
  star[WR].vinf = pin->GetReal("problem","vinf1");
  star[WR].col  = 1.0;
  // WR mass fractions
  star[WR].mass_frac[0] = pin->GetReal("problem","xH1");
  star[WR].mass_frac[1] = pin->GetReal("problem","xHe1");
  star[WR].mass_frac[2] = pin->GetReal("problem","xC1");
  star[WR].mass_frac[3] = pin->GetReal("problem","xN1");
  star[WR].mass_frac[4] = pin->GetReal("problem","xO1");
  // OB wind variables
  star[OB].mass = pin->GetReal("problem","mstar2");
  star[OB].mdot = pin->GetReal("problem","mdot2");
  star[OB].vinf = pin->GetReal("problem","vinf2");
  star[OB].col  = 0.0;
  // OB mass fractions
  star[OB].mass_frac[0] = pin->GetReal("problem","xH2");
  star[OB].mass_frac[1] = pin->GetReal("problem","xHe2");
  star[OB].mass_frac[2] = pin->GetReal("problem","xC2");
  star[OB].mass_frac[3] = pin->GetReal("problem","xN2");
  star[OB].mass_frac[4] = pin->GetReal("problem","xO2");
  // Other orbital properties
  ecc      = pin->GetReal("problem","ecc");
  period   = pin->GetReal("problem","period");
  phaseoff = pin->GetReal("problem","phaseoff");
  // Enable or disable simulation features
  cooling.enabled      = pin->GetBoolean("problem","cooling");
  dust.enabled         = pin->GetBoolean("problem","dust");
  dust_cooling.enabled = pin->GetBoolean("problem","dust_cooling");
  // Dust properties
  if (cooling.enabled) {
    cool_curve[WR].curve_file_name = pin->GetString("problem","coolcurve1");
    cool_curve[OB].curve_file_name = pin->GetString("problem","coolcurve2");
    cool_curve[WR].readPlasmaCoolingCurve();
    cool_curve[OB].readPlasmaCoolingCurve();
  }
  if (dust.enabled) {
    dust.z_init = pin->GetReal("problem","z_init");
    dust.a_init = pin->GetReal("problem","a_init");
  }
  if (dust_cooling.enabled) {
    ion_curve[WR].curve_file_name = pin->GetString("problem","ioncurve1");
    ion_curve[OB].curve_file_name = pin->GetString("problem","ioncurve2");
    ion_curve[WR].readIonCurve();
    ion_curve[OB].readIonCurve();
  }

  // Sanity checks, number of scalars and cartesians
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 1) {
    printf("!!! This problem series only works in Cartesian coordinates!\n");
    printf("!!! Please recompile with Cartesian coordinate system flag!\n");
    printf("!!! Exiting...\n");
    exit(EXIT_FAILURE);
  }
  if (NSCALARS == 0) {
    printf("!!! At least 1 scalar needs to be enabled (3 for dusty systems)!\n");
    printf("!!! Please recompile with the right number of scalars!\n");
    printf("!!! Exiting...\n");
    exit(EXIT_FAILURE);
  }
  if (dust.enabled) {
    if (NSCALARS < 3) {
      printf("!!! At least 3 scalars need to be available for dusty systems!\n");
      printf("!!! Please recompile with the right number of scalars!\n");
      printf("!!! Exiting...\n");
      exit(EXIT_FAILURE);
    }
  }
  
  // Check to see if dependent features are enabled
  if (Globals::my_rank == 0) {
    if (dust_cooling.enabled == true && cooling.enabled == false) {
      printf("!!! Error! Cooling needs to be enabled for dust cooling to be operational!\n");
      printf("!!! Exiting...\n");
      exit(EXIT_FAILURE);
    }
    if (dust_cooling.enabled == true && dust.enabled == false) {
      printf("!!! Error! Dust needs to be enabled for dust cooling to be operational!\n");
      printf("!!! Exiting...\n");
      exit(EXIT_FAILURE);
    }
  }

  // Now that files are read in, calculate or convert
  // Convert mass to CGS
  star[WR].mass *= MSOLTOGRAM;
  star[OB].mass *= MSOLTOGRAM;
  // Convert mass loss rate to CGS
  star[WR].mdot *= MSOLYRTOGS;
  star[OB].mdot *= MSOLYRTOGS;
  // Calculate momenta
  star[WR].calcWindMomentum();
  star[OB].calcWindMomentum();
  // Calculate wind momentum ratio
  wcr.eta = star[OB].mom / star[WR].mom;
  // Update frac_rob and frac_rwr
  wcr.calcFracs();
  // Calculate mass fractions for both stars using builtin class function
  star[WR].calcAvgMass();
  star[OB].calcAvgMass();
  
  // If simulation uses an adaptive mesh refinement code, configure AMR
  if (adaptive) {
    // Enroll AMR condition
    EnrollUserRefinementCondition(refinementCondition);
    // Active AMR object
    amr.enabled = true;
    // Determine resolution details
    // No mesh blocks exist, but we can extrapolate from athinput
    int blocksize_nx1 = pin->GetInteger("meshblock", "nx1");
    int blocksize_nx2 = pin->GetInteger("meshblock", "nx2");
    int blocksize_nx3 = pin->GetInteger("meshblock", "nx3");
    int nxBlocks = mesh_size.nx1/blocksize_nx1;
    int nyBlocks = mesh_size.nx2/blocksize_nx2;
    int nzBlocks = mesh_size.nx3/blocksize_nx3;
    int maxBlocks = std::max(nxBlocks,std::max(nyBlocks,nzBlocks));
    // Calculate min level and G0dx
    amr.G0_level = log2(maxBlocks);
    if (pow(2.0,amr.G0_level) < maxBlocks) amr.G0_level++;
    Real x1size = mesh_size.x1max - mesh_size.x1min;
    amr.G0dx = x1size / mesh_size.nx1;
  }

  // Enroll the explicit source function
  EnrollUserExplicitSourceFunction(physicalSources);

  // Calculate orbit for the first time
  orbitCalc(time);

  // Set stdout precision
  std::cout.precision(3);       // Set to 3 decimal places
  std::cout << std::scientific; // Set to use scientific notation for all outputs if cout called
  std::cerr << std::scientific; // Ditto for cerr
  // Setup complete, print out variables
  if (Globals::my_rank ==0) {
    printf("!!! Setup complete!\n");

    printf("> Features\n");
    printf(">  Cooling:      %s\n", cooling.enabled ? "Enabled" : "Disabled");
    printf(">  Dust:         %s\n", dust.enabled ? "Enabled" : "Disabled");
    printf(">  Dust cooling: %s\n", dust_cooling.enabled ? "Enabled" : "Disabled");
    printf(">  AMR:          %s\n", amr.enabled ? "Enabled" : "Disabled");

    printf("> Star properties\n");
    printf(">  Star masses:  %.3e %.3e g\n",star[WR].mass,star[OB].mass);
    printf(">  Mass loss:    %.3e %.3e g/s\n",star[WR].mdot,star[OB].mdot);
    printf(">  Terminal vel: %.3e %.3e cm/s\n",star[WR].vinf,star[OB].vinf);

    printf("> Orbital properties\n");
    printf(">  Period:   %.3e \n",period);
    printf(">  Phaseoff: %.3f \n",phaseoff);
    printf(">  Ecc:      %.3f \n",ecc);
    printf(">  X:        %+.3e %+.3e cm\n",star[WR].pos[0],star[OB].pos[0]);
    printf(">  Y:        %+.3e %+.3e cm\n",star[WR].pos[1],star[OB].pos[1]);
    printf(">  Z:        %+.3e %+.3e cm\n",star[WR].pos[2],star[OB].pos[2]);
    printf(">  d_sep:    %.3e cm\n",wcr.d_sep);
    printf(">  r_wcr:    %+.3e %+.3e cm\n",wcr.r_wr,wcr.r_ob);
    printf(">  stag_pos: %+.3e %+.3e %+.3e cm\n",wcr.pos[0],wcr.pos[1],wcr.pos[2]);
    
    printf("> Abundances\n");
    printf(">  xH:   %.3f %.3f\n",star[WR].mass_frac[0],star[OB].mass_frac[0]);
    printf(">  xHe:  %.3f %.3f\n",star[WR].mass_frac[1],star[OB].mass_frac[1]);
    printf(">  xC:   %.3f %.3f\n",star[WR].mass_frac[2],star[OB].mass_frac[2]);
    printf(">  xN:   %.3f %.3f\n",star[WR].mass_frac[3],star[OB].mass_frac[3]);
    printf(">  xO:   %.3f %.3f\n",star[WR].mass_frac[4],star[OB].mass_frac[4]);
    printf(">  Mu:   %.3f %.3f\n",star[WR].mu,star[OB].mu);
    printf(">  AvgM: %.3e %.3e g\n",star[WR].avg_mass,star[OB].avg_mass);

    if (dust.enabled) {
      printf("> Dust properties\n");
      printf(">  a_init: %.3e\n",dust.a_init);
      printf(">  z_init: %.3e\n",dust.z_init);
    }

    if (cooling.enabled) {
      printf("> Cooling parameters\n");
      std::cout << ">  WR Filename: " << cool_curve[WR].curve_file_name << "\n";
      std::cout << ">  OB Filename: " << cool_curve[OB].curve_file_name << "\n";
    }

    if (dust_cooling.enabled) {
      printf("> Dust cooling parameters\n");
      std::cout << ">  WR Filename: " << ion_curve[WR].curve_file_name << "\n";
      std::cout << ">  OB Filename: " << ion_curve[OB].curve_file_name << "\n";
    }
  
    printf("!!! Starting procesing now!\n");
  }
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Colliding Wind Binary problem generator
//!        Initialise winds and zero out scalars that are being used
//!        Winds are initialised by finding the stagnation point along
//!        the x axis and dividing the simulation into two regions
//!        initially dominated by each wind, a smooth wind function
//!        then maps on initial density and pressure.
//!        The regions are divided by testing to see which side of the stagnation point
//!        the cell is on.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get heat capacity ratio
  Real gmma  = peos->GetGamma();
  Real gmma1 = gmma - 1.0;
  // Loop over all cells with both winds, wind in begening of
  for (int n = 0; n < NWIND; n++) {
    for (int k = ks; k <= ke; k++) {
      // Calculate z component of distance and d**2 from star
      Real zc   = pcoord->x3v(k) - star[n].pos[2];
      Real zc2  = SQR(zc);
      for (int j = js; j <= je; j++) {
        // Calculate y component of distance and d**2 from star
        Real yc   = pcoord->x2v(j) - star[n].pos[1];
        Real yc2  = SQR(yc);
        for (int i = is; i <= ie; i++) {
          // Calculate x component of distance and d**2 from star
          Real xc   = pcoord->x1v(i) - star[n].pos[0];
          Real xc2  = SQR(xc);
          // Calculate distance from star
          Real r2 = xc2 + yc2 + zc2;
          Real r  = sqrt(r2);
          // Calculate xy distance from star
          Real xy2 = xc2 + yc2;
          Real xy  = sqrt(xy2);
          // Calculate angular components
          Real sin_phi   = xy/r;
          Real cos_phi   = zc/r;
          Real sin_theta = yc/xy;
          Real cos_theta = xc/xy;
          // Determine if cell is on the same side of the WCR as the star being tested
          Real xx = pcoord->x1v(i);
          Real wcr_star_x = star[n].pos[0] - wcr.pos[0];  
          Real wcr_cell_x = xx - wcr.pos[0];
          // Multiply together, if value is positive, cell is on the same side of the WCR as the star it represents, if otherwise, then it is not
          Real star_cell_test = wcr_star_x * wcr_cell_x;
          if (star_cell_test > 0.0) {
            // Calculate density
            Real rho = star[n].mdot / (4.0*PI*r2*star[n].vinf);
            // Calculate momentum components of initial wind
            Real vel = star[n].vinf;
            Real u1  = vel*sin_phi*cos_theta;
                 u1 += star[n].vel[0];
            Real m1  = u1 * rho;
            Real u2  = vel*sin_phi*sin_theta;
                 u2 += star[n].vel[1];
            Real m2  = u2 * rho;
            Real u3  = vel*cos_phi;
            Real m3  = u3 * rho;
            // Calculate pressure
            Real pre = (rho/star[n].avg_mass) * KBOLTZ * star[n].Twind;
            // Calculate total energy in cell
            Real e_int = pre / gmma1;
            Real e_kin = 0.5 * rho * (SQR(u1) + SQR(u2) + SQR(u3));
            Real e_tot = e_int + e_kin;
            // Rewrite cell conserved variables
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = m1;
            phydro->u(IM2,k,j,i) = m2;
            phydro->u(IM3,k,j,i) = m3;
            phydro->u(IEN,k,j,i) = e_tot;
            // Rewrite colour scalar
            pscalars->s(CLOC,k,j,i) = star[n].col * rho;
            // Blank all other scalars
            for (int nscal = 1; nscal < NSCALARS; nscal++) {
              pscalars->s(nscal,k,j,i) = 0.0;
            }
          }
        }
      }
    }
  }
  // Finished!
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once after every time step for user-defined work.
//         Because time has not yet been updated we need to add on dt.
//         This is used to update the stellar positions, dsep, etc.
//         Otherwise, most processing is handled in meshblocks! 
//========================================================================================
void Mesh::UserWorkInLoop() {
  orbitCalc(time+dt);
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop(ParameterInput *pin)
//! \brief Map on winds and constrain passive scalars that require constraining
//! Function is used to simulate outflows for stars, this is performed by fixing the
//! density, energy and momentum values around the cartesian coordinates of the star
//! Dust is injected into the simulation here, using initial small grains with a very
//! limited amount.
//! This requires that stars are adequately resolved, otherwise winds may come out wrong
//! ideally a minimum of 6 cells at the finest resolution from the orbital position of 
//! the sta.
//! Potential optimisations:
//!  Since the block consists of evenly spaced cells, it should be easy to rule out
//!  blocks that are clearly not within the remap zone, reducing the number of radius 
//!  tests.
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  // Get heat capacity ratio
  Real gmma  = peos->GetGamma();
  Real gmma1 = gmma - 1.0;
  // Get timestep, all meshblocks operate synchronously
  Real dt = pmy_mesh->dt;
  // Begin remapping winds
  for (int n = 0; n < NWIND; n++) {
    Real dx = pcoord->dx1v(0); // Cell width for this meshblock
    // Calculate size of remap radius
    Real remap_radius = star[n].ncells_remap * dx; // Calculate remap radius
    // Remap winds for star n
    for (int k = ks; k <= ke; k++) {
      // Calculate z component of distance and d**2 from star
      Real zc  = pcoord->x3v(k) - star[n].pos[2];
      Real zc2 = SQR(zc);
      for (int j = js; j <= je; j++) {
        // Calculate y component of distance and d**2 from star
        Real yc  = pcoord->x2v(j) - star[n].pos[1];
        Real yc2 = SQR(yc);
        for (int i = is; i <= ie; i++) {
          // Calculate x component of distance and d**2 from star
          Real xc  = pcoord->x1v(i) - star[n].pos[0];
          Real xc2 = SQR(xc);
          // Calculate distance from star
          Real r2 = xc2 + yc2 + zc2;
          Real r  = sqrt(r2);
          // Check to cell if cell is within the remap region
          if (r < remap_radius) {
            // Calculate xy distance from star
            Real xy2 = xc2 + yc2;
            Real xy  = sqrt(xy2);
            // Calculate angular components
            Real sin_phi   = xy/r;
            Real cos_phi   = zc/r;
            Real sin_theta = yc/xy;
            Real cos_theta = xc/xy;
            // Define physical constants
            Real rho = star[n].mdot / (4.0*PI*r2*star[n].vinf);
            // Calculate momentum components of initial wind
            Real vel = star[n].vinf;
            Real u1  = vel*sin_phi*cos_theta;  // Calculate 
                 u1 += star[n].vel[0];         // Add orbital velocity component to wind
            Real m1  = u1 * rho;               // Calculate X momentum component
            Real u2  = vel*sin_phi*sin_theta;
                 u2 += star[n].vel[1];
            Real m2  = u2 * rho;
            Real u3  = vel*cos_phi;
            Real m3  = u3 * rho;
            // Calculate pressure
            Real pre = (rho/star[n].avg_mass) * KBOLTZ * star[n].Twind;
            // Calculate total energy in cell
            Real e_int = pre / gmma1;
            Real e_kin = 0.5 * rho * (SQR(u1) + SQR(u2) + SQR(u3));
            Real e_tot = e_int + e_kin;
            // Rewrite cell conserved variables
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = m1;
            phydro->u(IM2,k,j,i) = m2;
            phydro->u(IM3,k,j,i) = m3;
            phydro->u(IEN,k,j,i) = e_tot;
            // Rewrite passive scalars
            Real c_density = star[n].col * rho;
            pscalars->s(CLOC,k,j,i) = c_density;
            if (dust.enabled) {
              Real z_density = dust.z_init * rho;
              Real a_density = dust.a_init * rho;
              pscalars->s(ZLOC,k,j,i) = z_density;
              pscalars->s(ALOC,k,j,i) = a_density;
            }
          }
        }
      }
    }
  }

  // Second run through MeshBlock, limit wind colour and set level scalar
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real rho = phydro->u(IDN,k,j,i);
        Real col = pscalars->s(CLOC,k,j,i);
        if (col > rho) {
          col = rho;
        }
        if (col < 0.0) {
          col = 0.0;
        }
        pscalars->s(CLOC,k,j,i) = col;
        pscalars->s(LLOC,k,j,i) = loc.level * rho;
      }
    }
  }

  // Finished!
  return;
}


//========================================================================================
//! \fn int refinementCondition(ParameterInput *pmb)
//! \brief Refinement conditions for problem, if AMR is enabled, these will be enabled
//!  Function returns one of 3 values:
//!  -1: Flag block for de-refinement, this does not immediately de-refine the cell, but
//!      refines after a defined number of steps, and if surrounding cells are flagged
//!      for de-refinement
//!   0: Do absolutely nothing
//!   1: Refine cell on next timestep
//!  How this function works:
//!    The main method is to refine near individual stars, as well as around the wind
//!    stagnation point, will refine to maximum simulation level.
//!    This is based on the number of cells distance, this means that refinement should
//!    be continuous and smooth outwards from the 3 refinement points
//!  See https://github.com/PrincetonUniversity/athena/wiki/Adaptive-Mesh-Refinement
//========================================================================================

int refinementCondition(MeshBlock *pmb) {
  // First, get current cell level width, this assumes that all cells are square (they should be for this problem!)
  Real dx = pmb->pcoord->dx1v(0);
  // Check to see if meshblock contains stars, refine around region to max level
  for (int n = 0; n < NWIND; n++) {
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      // Get Z co-ordinate separation from star
      Real zc  = pmb->pcoord->x3v(k) - star[n].pos[2];
      Real zc2 = SQR(zc);
      for (int j = pmb->js; j <= pmb->je; j++) {
        // Get Y co-ordinate separation from star
        Real yc  = pmb->pcoord->x2v(j) - star[n].pos[1];
        Real yc2 = SQR(yc);
        for (int i = pmb->is; i <= pmb->ie; i++) {
          // Get X co-ordinate separation from star
          Real xc  = pmb->pcoord->x1v(i) - star[n].pos[0];
          Real xc2 = SQR(xc);
          // Get radial distance from current star to current cell
          Real r2 = xc2 + yc2 + zc2;
          Real r  = sqrt(r2);
          // Get approximate number of cells between current cell and current star
          int ri = int(r/dx);
          // If number of cells distance is below threshold, refine until maximum level reached
          if (ri < amr.star_separation) {
            return 1;
          }
        }
      }
    }
  }

  // Check to see if meshblock contains the stagnation point
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    // Get Z co-ordinate separation from stagnation point
    Real zc  = pmb->pcoord->x3v(k) - wcr.pos[2];
    Real zc2 = SQR(zc);
    for (int j = pmb->js; j <= pmb->je; j++) {
      // Get Y co-ordinate separation from stagnation point
      Real yc  = pmb->pcoord->x2v(j) - wcr.pos[1];
      Real yc2 = SQR(yc);
      for (int i = pmb->is; i <= pmb->ie; i++) {
        // Get X co-ordinate separation from stagnation point
        Real xc  = pmb->pcoord->x1v(i) - wcr.pos[0];
        Real xc2 = SQR(xc);
        // Get radial distnace from WCR to current cell
        Real r2 = xc2 + yc2 + zc2;
        Real r  = sqrt(r2);
        // Get approximate number of cells between current cell and WCR
        int ri = int(r/dx);
        // If number of cells distance is below threshold, refine until maximum level reached
        if (ri < amr.wcr_separation) {
          return 1;
        }
      }
    }
  }

  // If the function has reached this point, meshblock can be flagged for de-refinement
  return -1;
}

//========================================================================================
//! \fn physicalSources
//! \brief Function enrolled into meshblock class that calls other functions
//! Functions are called from this function, since only one function can be enrolled at 
//! a time. The functions enabled are depdendent on the simulation feature set
void physicalSources(MeshBlock *pmb, const Real time, const Real dt,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                     AthenaArray<Real> &cons_scalar) {
  
  if (cooling.enabled) {
    radiateApprox(pmb,dt,cons);
  }
  if (dust.enabled) {
    EvolveDustMultiWind(pmb,dt,cons);
  }
  return;
}

// All functions not related to Mesh:: or MeshBlock:: class below this line
// =======================================================================================

//========================================================================================
//! \fn radiateApprox
//! \brief Calculate cooling of gas by approximating plasma and dust cooling
//! Simulation requires fast and reasonably accurate calculation of energy energy loss due
//! to physical proccesses such as dust heating, H recombination etc.
//! This function approximates energy loss due to plasma processes by using a lookup table
//! interpolation, and scaling the energy loss relative to the particle number density in
//! the cell. 
//! Dust cooling is approximated using the formulae described in Dwek & Werner 1981, an 
//! approximation of electron opacity is utilised which is at worst 10% off, but 5 orders
//! of magnitude faster than an integral method.
//! Additional functions are used to smooth out cooling between unresolved meshblock 
//! edges and to adjust pressure safetly.
//========================================================================================
void radiateApprox(MeshBlock *pmb, const Real dt,
                   AthenaArray<Real> &cons) {
  // Get constants
  const Real gmma  = pmb->peos->GetGamma();
  const Real gmma1 = gmma - 1.0;
  // Create array to store change in internal energy
  AthenaArray<Real> d_e_int(pmb->ke+1,pmb->je+1,pmb->ie+1);
  AthenaArray<Real> avgmass(pmb->ke+1,pmb->je+1,pmb->ie+1);
  // Loop through all cells
  for (int k = pmb->ks; k <= pmb->ke; ++k){
    for (int j = pmb->js; j <= pmb->je; ++j){
      for (int i = pmb->is; i <= pmb->ie; ++i){
        // Initialise cell
        // Get gas variables
        Real rho   = cons(IDN,k,j,i);        // Density (g/cm^3)
        Real u1    = cons(IM1,k,j,i) / rho;  // Wind velocity X component (cm/s)
        Real u2    = cons(IM2,k,j,i) / rho;  // Wind velocity Y component (cm/s)
        Real u3    = cons(IM3,k,j,i) / rho;  // Wind velocity Z component (cm/s)
        Real e_tot = cons(IEN,k,j,i);        // Total energy (erg)
        // Get scalars
        Real col = pmb->pscalars->s(CLOC,k,j,i) / rho;  // Wind colour
        // Initialise dust scalars
        Real z    = 0.0;  // Dust-to-gas mass ratio
        Real a    = 0.0;  // Grain radius (micron)
        Real a_cm = 0.0;  // Grain radius (cm)
        // Initialise dust parameters
        Real rho_D;    // Dust density (g/cm^3)
        Real vol_Gr;   // Grain volume (cm^3)
        Real mass_Gr;  // Grain mass (g)
        Real nD;       // Grain number density (1/cm^3)
        if (dust_cooling.enabled) {
          z       = pmb->pscalars->s(ZLOC,k,j,i) / rho;
          a       = pmb->pscalars->s(ALOC,k,j,i) / rho;  // Get grain radius in microns
          a_cm    = a * MICRONTOCM;
          rho_D   = rho * z;
          vol_Gr  = (4.0/3.0) * PI * CUBE(a_cm);
          mass_Gr = dust.rho_Gr * vol_Gr;
          // Finish up nD calculation
          nD = rho_D / mass_Gr;
        }
        // Calculate gas kinetic energy
        Real v2    = SQR(u1) + SQR(u2) + SQR(u3);  // Gas velocity squared (cm^2/s^2)
        Real e_kin = 0.5 * rho * v2;               // Kinetic energy (erg)
        // Calculate internal energy
        Real e_int = e_tot - e_kin;  // Internal energy (erg)
        Real pre   = gmma1 * e_int;  // Pressure (Barye)
        // Calculate average mass of particle in cell
        Real cols[2];
             cols[WR] = col;
             cols[OB] = 1.0 - col;
        Real avg_mass = (cols[WR] * star[WR].avg_mass) + (cols[OB] * star[OB].avg_mass);
        // Calculate temperature 
        Real T = (pre * avg_mass) / (rho * KBOLTZ);  // Mixed gas temperature (K)
        // Store initial temp in its own variable
        Real T_i = T;  // Initial temperature (K)
        // Calculate hydrogen number density for cooling curves
        Real nH = rho / MASSH; // Hydrogen number density (1/cm^3)
        // Finished initialisation
        // Prevent time-wasting in pure winds
        if (T < 1.1 * star[WR].Twind && cols[WR] == 1.0) {
          T = star[WR].Twind;
        }
        else if (T < 1.1 * star[OB].Twind && cols[OB] == 1.0) {
          T = star[OB].Twind;
        }
        // Otherwise, begin the actual cooling loop!
        else {
          Real dt_int = 0.0; // Current substep integrated timestep
          while (dt_int < dt) {
            Real dE_dt_tot = 0.0;
            // PLASMA COOLING
            // Calculate lambda from linear interpolation of lookup table
            Real lambda_gas = 0.0;  // Gas cooling parameter
            for (int n = 0; n < NWIND; n++) {
              // Find and interpolate an appropriate value for lambda
              Real lambda_wind = searchAndInterpolate(cool_curve[n].T,
                                                      cool_curve[n].L,
                                                      T);
              // Calculate density weighted lambda
              Real lambda_wind_frac = lambda_wind * cols[n];
              // Add to lambda_gas to calculate value for lambda
              lambda_gas += lambda_wind_frac;
            }
            Real dE_dt_gas  = SQR(nH) * lambda_gas;
                 dE_dt_tot += dE_dt_gas;
            // DUST COOLING
            if (dust_cooling.enabled) {
              Real dE_dt_dust = 0.0;
              for (int n = 0; n < NWIND; n++) {
                // Calculate heating rate for a single grain
                Real dE_dt_grain      = calcGrainCoolRate(rho,a,T,star[n],ion_curve[n]);
                Real dE_dt_dust_wind  = nD * dE_dt_grain;  // Calculate total heating rate for wind
                     dE_dt_dust_wind *= cols[n];           // Scale based on wind contribution
                // Add to total 
                dE_dt_dust += dE_dt_dust_wind;
              }
              // Add cooling rate from dust to total
              dE_dt_tot += dE_dt_dust;
            }
            // FINISH UP SUBSTEP
            Real E_int_new = pre / gmma1;            // Find current internal energy
            Real t_cool    = E_int_new / dE_dt_tot;  // Cooling time (s)
            // Calculate maximum timestep to sample the cooling curve
            Real dt_cool = 0.1 * fabs(t_cool);  // Maximum timestep
            // Check to see if current integration attempt will overrun, adjust if it is
            if ((dt_int + dt_cool) > dt) {
              dt_cool = dt - dt_int;
            }
            dt_int += dt_cool;
            // Calculate new temperature, update pressure
            Real dE = -dE_dt_tot * dt_cool;
            Real T_new = T * (E_int_new * dE) / E_int_new;
            // Check to see if cooling needs to exit
            if (T_new < cool_curve[WR].T_min) {
              T = cool_curve[WR].T_min;
              break;
            }
            if (T_new < cool_curve[OB].T_min) {
              T = cool_curve[OB].T_min;
              break;
            }
            // Update pressure and current temperature
            pre *= T_new / T;
            T    = T_new;
          }
        }
        // End of cooling loop for cell
        Real T_f = T;
        // Write change in energy and cell average mass to array
        // For processing in adjustPressureDueToCooling()
        d_e_int(k,j,i) = (T_i - T_f) * rho;
        avgmass(k,j,i) = avg_mass;
      }
    }
  }

  // Restrict cooling at unresolved interfaces
  restrictCool(pmb->is,pmb->ie,
               pmb->js,pmb->je,
               pmb->ks,pmb->ke,
               pmb->pmy_mesh->ndim,
               gmma1,
               d_e_int,
               cons);

  // Adjust pressure due to cooling
  adjustPressureDueToCooling(pmb->is,pmb->ie,
                             pmb->js,pmb->je,
                             pmb->ks,pmb->ke,
                             gmma1,
                             d_e_int,
                             avgmass,
                             cons);

  // Finished!
  return;
}

//! \fn Real CalcLambdaDust(Real nH, Real a, Real T)
//  \brief Calculate energy loss per dust grain, multiply by nD to calculate
//         cell cooling rate
//  - Energy lost from the gas flow due to dust is mainly due to
//    collisional heating of the dust particles from atoms and
//    electrons
//  - Efficiency losses can occur at high temperatures as particles
//           are so energetic they pass through one another
//  - This function approximates this effect
//  - Resultant value for single grain, to find energy loss in erg/s/cm^3
//    value must be multiplied by nD
//         Derived from:
//         Dwek, E., & Werner, M. W. (1981).
//         The Infrared Emission From Supernova Condensates.
//         The Astrophysical Journal, 248, 138.
//         https://doi.org/10.1086/159138

Real calcGrainCoolRate(Real rho_G, Real a, Real T, Star star, IonCurve ion_curve) {
  // Shorten kBT, since used a lot
  const Real kBT = T * KBOLTZ;
  // Use precomputed arrays for speed
  // Precalculated arrays for critical energy constant and atomic mass
  // in CGS units for each type of element are declared:
  //   - For H atoms:   Ec = 133keV, m = 1.0 * MASSH
  //   - For He atoms:  Ec = 222keV, m = 4.0 * MASSH
  //   - For CNO atoms: Ec = 665keV, m = 12.0,14.0,16.0 * MASSH
  const Real E_E[5] = {2.1308949e-07,3.5568321e-07,1.0654475e-06,1.0654475e-06,1.0654475e-06};
  const Real m_E[5] = {1.6735575e-24,6.6942300e-24,2.0082699e-23,2.3429805e-23,2.6776920e-23};
  // Storage arrays for values
  Real n_E[5] = {0.0,0.0,0.0,0.0,0.0};  // Elemental number density, cm^-3
  Real H_E[5] = {0.0,0.0,0.0,0.0,0.0};  // Heating for each element, erg s^-1
  // Other variables
  Real n_T    = 0.0;
  Real H_coll = 0.0;

  for (int n = 0; n < 5; n++) {
    // Calculate number density for element
    n_E[n] = rho_G * star.norm_n_E[n];
    // Calculate total number density
    n_T += n_E[n];
    // Calculate the critical energy of incident hydrogen atoms
    Real EC = E_E[n] * a;  
    // Calculate grain heating effciency due to atoms
    Real h_n = 1.0 - (1.0 + EC / (2.0 * kBT)) * exp(-EC/kBT);
    // Calculate heating rate of element, Eq 2 of DW81
    H_E[n]  = 1.26e-19 * SQR(a) * pow(T,1.5) * n_E[n] * h_n;
    H_E[n] /= sqrt(m_E[n]/MASSH);
    H_coll += H_E[n];  // Add to counter
  }

  // Calculate contribution due to electron heating

  // First, find number of free electron, using ionisation lookup table
  Real nFreeElectrons = 0.0;
  if (T < ion_curve.T_min) {
    nFreeElectrons = std::min(1.0,ion_curve.E_min);
  }
  else if (T > ion_curve.T_max) {
    nFreeElectrons = ion_curve.E_max;
  }
  else {
    nFreeElectrons = searchAndInterpolate(ion_curve.T,ion_curve.E,T);
  }
  // Using this estimated value, calculate the electron number density
  Real n_e = n_T * nFreeElectrons;
  // Calculate the critical energy for electron to penetrate grain
  // This makes the assumption of an uncharged dust grain, Ee = Ec
  Real Ee  = 3.6850063e-08 * pow(a,2.0/3.0);  // DW81 Eq A6
  Real x_e = Ee/kBT;                          // DW81 Eq A11
  // Approximate electron-grain "transparency"
  Real h_e = 0.0;
  if (x_e > 4.5) {
    h_e = 1;
  }
  else if (x_e > 1.5) {
    h_e = 0.37 * pow(x_e,0.62);
  }
  else {
    h_e = 0.27 * pow(x_e,1.50);
  }
  // Calculate heating rate due to electrons
  Real H_el  = 1.26e-19 * SQR(a) * pow(T,1.5) * n_e * h_e;
       H_el /= sqrt(MASSE/MASSH);
  // Summate heating rates and normalise by nd*np
  Real edotGrain = (H_coll + H_el);
  // Finish up and return!
  return edotGrain;
}

//!  \brief Restrict the cooling rate at unresolved interfaces between hot 
//!         diffuse gas and cold dense gas.
//!
//!  Replace deltaE with minimum of neighboring deltaEs at the interface.
//!  Updates dei, which is positive if the gas is cooling.
//!
//!  \author Julian Pittard (Original version 13.09.11)
//!  \version 1.0-stable (Evenstar):
//!  \date Last modified: 13.09.11 (JMP)
void restrictCool(int is,int ie,
                  int js,int je,
                  int ks,int ke,
                  int nd,
                  Real gmma1,
                  AthenaArray<Real> &dei,
                  const AthenaArray<Real> &cons) {
  AthenaArray<Real> pre(ke+1,je+1,ie+1);
  AthenaArray<Real> scrch(ie+1), dis(ie+1), drhox(ie+1), drhoy(ie+1), drhoz(ie+1);
  
  for (int k = ks; k <= ke; k++){
    for (int j = js; j <= je; j++){
      for (int i = is; i <= ie; i++){
        Real rho = cons(IDN,k,j,i);
	      Real u1  = cons(IM1,k,j,i)/rho; 
	      Real u2  = cons(IM2,k,j,i)/rho; 
	      Real u3  = cons(IM3,k,j,i)/rho;
	      Real ke = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
	      Real ie = cons(IEN,k,j,i) - ke;
	      pre(k,j,i) = gmma1*ie;
      }
    }
  }
  
  for (int k = ks; k <= ke; k++){
    int ktp = std::min(k+1,ke);
    int kbt = std::max(k-1,ks);
    for (int j = js; j <= je; j++){
      // Locate cloud interfaces => dis = 1, otherwise, dis = -1
      int jtp = std::min(j+1,je);
      int jbt = std::max(j-1,js);
      for (int i = is+1; i < ie; i++){
	      scrch(i) = std::min(cons(IDN,k,j,i+1),cons(IDN,k,j,i-1));
	      drhox(i) = cons(IDN,k,j,i+1) - cons(IDN,k,j,i-1);
        drhox(i) = std::copysign(drhox(i),(pre(k,j,i-1)/cons(IDN,k,j,i-1)
			     - pre(k,j,i+1)/cons(IDN,k,j,i+1)))/scrch(i) - 2;
	      if (nd > 1){
  	      scrch(i) = std::min(cons(IDN,k,jtp,i),cons(IDN,k,jbt,i));
	        drhoy(i) = cons(IDN,k,jtp,i) - cons(IDN,k,jbt,i);
          drhoy(i) = std::copysign(drhoy(i),(pre(k,jbt,i)/cons(IDN,k,jbt,i)
			     - pre(k,jtp,i)/cons(IDN,k,jtp,i)))/scrch(i) - 2;
        }
	      if (nd == 3){
  	      scrch(i) = std::min(cons(IDN,ktp,j,i),cons(IDN,kbt,j,i));
	        drhoz(i) = cons(IDN,ktp,j,i) - cons(IDN,kbt,j,i);
          drhoz(i) = std::copysign(drhoz(i),(pre(kbt,j,i)/cons(IDN,kbt,j,i)
			     - pre(ktp,j,i)/cons(IDN,ktp,j,i)))/scrch(i) - 2;
        }
      	if      (nd == 1) dis(i) = drhox(i);
	      else if (nd == 2) dis(i) = std::max(drhox(i),drhoy(i));
	      else              dis(i) = std::max(drhox(i),std::max(drhoy(i),drhoz(i)));
      }
      dis(is) = -1.0;
      dis(ie) = -1.0;

      for (int i = is; i <= ie; i++){
	      int itp = std::min(i+1,ie);
	      int ibt = std::max(i-1,is);
	      if (dis(i) > 0.0){
  	      if      (nd == 1) scrch(i) = std::min(dei(k,j,ibt),dei(k,j,itp));
	        else if (nd == 2) scrch(i) = std::min(dei(k,j,ibt),std::min(dei(k,j,itp),
					  std::min(dei(k,jtp,i),dei(k,jbt,i))));
          else              scrch(i) = std::min(dei(k,j,ibt),std::min(dei(k,j,itp),
				     std::min(dei(k,jtp,i),std::min(dei(k,jbt,i),
				     std::min(dei(ktp,j,i),dei(kbt,j,i))))));
	        //std::cout << "k = " << k << "; j = " << j << "; i = " << i << "; scrch = " << scrch(i) << "; dei = " << dei(k,j,i) << "\n";
	        //exit(EXIT_SUCCESS);
	        dei(k,j,i) = scrch(i);
	      }
      }

    } // j loop
  } // k loop
  
  return;
}

/*!  \brief Adjust the pressure due to cooling.
 *
 *   Uses dei().
 *
 *   \author Julian Pittard (Original version 13.09.11)
 *   \version 1.0-stable (Evenstar):
 *   \date Last modified: 13.09.11 (JMP)
 */
void adjustPressureDueToCooling(int is,int ie,int js,int je,int ks,int ke,Real gmma1,AthenaArray<Real> &dei,AthenaArray<Real> &avgm,AthenaArray<Real> &cons){
  Real tmin = 1e4;
  Real tmax = 3e9;
  for (int k = ks; k <= ke; k++){
    for (int j = js; j <= je; j++){
      for (int i = is; i <= ie; i++){
        Real rho      = cons(IDN,k,j,i);
        Real u1       = cons(IM1,k,j,i)/rho;
        Real u2       = cons(IM2,k,j,i)/rho;
        Real u3       = cons(IM3,k,j,i)/rho;
        Real ke       = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
        Real pre      = (cons(IEN, k, j, i) - ke)*gmma1;
        Real avg_mass = avgm(k,j,i);
        Real const_1  = avg_mass / KBOLTZ;
        Real tmpold   = const_1 * pre / rho;
	      Real tmpnew;
        if (std::isnan(tmpold) || std::isinf(tmpold)){
	        // Set temperature to Tmin
	        tmpnew = tmin;
        }
  	    else{
	        Real dtemp = dei(k,j,i)/rho;
	        tmpnew = std::max((tmpold-dtemp),tmin);
	        tmpnew = std::min(tmpnew,tmax);
          if (std::isnan(tmpnew) || std::isinf(tmpnew)){
	          tmpnew = tmin;
	        }
	      }
        Real pnew = tmpnew * rho / const_1;
        // Update conserved values
        cons(IEN, k, j, i) = ke + pnew / gmma1;

      } // i loop
    }   // j loop
  }     // k loop
  return;
}

//! \fn void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons)
/*! \brief Grow and shrink dust grains in accordance with growth and destruction mechanisms
 *
 *  Evolve dust, applying growth and destruction mechanisms
 */
void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons) {
  // Get gamma for simulation
  const Real gmma  = pmb->peos->GetGamma();
  const Real gmma1 = gmma - 1.0;
  
  // Loop through cells in MeshBlock
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      for (int i = pmb->is; i <= pmb->ie; i++) {
        // Import conserved variables
        Real rho   = cons(IDN,k,j,i);        // Dust density
        Real e_tot = cons(IEN,k,j,i);        // Total energy (erg)
        Real u1    = cons(IM1,k,j,i) / rho;  // Gas velocity X component (cm/s)
        Real u2    = cons(IM2,k,j,i) / rho;  // Gas velocity Y component (cm/s)
        Real u3    = cons(IM3,k,j,i) / rho;  // Gas velocity X component (cm/s)
        // Import scalars
        Real col = pmb->pscalars->s(CLOC,k,j,i) / rho;  // Wind colour 
        Real z   = pmb->pscalars->s(ALOC,k,j,i) / rho;  // Dust mass fraction
        Real a   = pmb->pscalars->s(ALOC,k,j,i) / rho;  // Grain radius (micron)
             a  *= MICRONTOCM;                          // Convert grain radius to CM
        // Calculate gas properties
        Real v2    = SQR(u1) + SQR(u2) + SQR(u3);  // Velocity**2 (cm^2/s^2)
        Real e_kin = 0.5 * rho * v2;               // Kinetic energy (erg)
        Real pre   = (e_tot - e_kin) * gmma1;      // Gas pressure (Ba)
        // Calculate wind abundances
        Real cols[2] = {0.0,0.0};
             cols[0] = col;
             cols[1] = 1.0 - col;
        Real C_mass_frac = 0.0;
        for (int n = 0; n < NWIND; n++) {
          Real carbon_abundance = star[n].mass_frac[2];
          Real wind_C_mass_frac = cols[n] * carbon_abundance;
          C_mass_frac += wind_C_mass_frac;
        }
        Real rho_C = rho * C_mass_frac; // Gas density of Carbon element in wind (g/cm^3)
        // Calculate average particle mass for gas, averaged between winds (g)
        Real avg_mass = (cols[WR] * star[WR].avg_mass) * (cols[OB] * star[OB].avg_mass);
        // Calculate gas temperature
        Real T = (avg_mass * pre) / (rho * KBOLTZ); // Gas temperature (K)
        // Calculate number density of wind
        Real n_T = rho / avg_mass;
        // Calculate dust grain parameters
        Real rho_D   = rho * z;                                 // Dust density (g/cm^3)
        Real mass_Gr = (4.0/3.0) * PI * CUBE(a) * dust.rho_Gr;  // Dust grain mass (g)
        Real n_D     = rho_D / mass_Gr;                         // Dust number density (1/cm^3)
        // Initialise grain growth and destruction variables
        Real drho_D_dt = 0.0;  // Change in dust density (g/cm^3/s)
        Real da_dt     = 0.0;  // Change in grain radius (cm/s)
        // Finished initialisation, compute grain growth and destruction
        if (z > dust.z_min && a > dust.a_min) {
          if (T > 1.0e6) {
            Real tau_D = (3.156e17 * a) / n_T;  // Grain destruction time (s)
            da_dt      = -a / tau_D;
            drho_D_dt  = -1.33e-17 * dust.rho_Gr * SQR(a) * n_T * n_D; 
          }
          else if (T < 1.4e4) {
            Real wa   = sqrt((3.0 * KBOLTZ * T) / (12.0 * MASSH));  // Velocity of carbon atoms
            da_dt     = (0.25 * dust.eps_a * rho_C* wa) / dust.rho_Gr;
            drho_D_dt = 4.0 * PI * SQR(a) * dust.rho_Gr * n_D * da_dt;
          }
          // If there is any change in growth, calculate changes to conserved variables and scalars
          if (drho_D_dt != 0.0) {
            // Calculate new dust density
            Real min_rho_D = rho * dust.z_min;
            Real drho_D    = drho_D_dt * dt;
            Real rho_D_new = rho_D + drho_D;
                 rho_D_new = std::max(min_rho_D,rho_D_new);
            // Calculate new gas density
            Real rho_new = rho + (rho_D - rho_D_new);
            // Calculate new dust-to-gas mass ratio 
            Real z_new = rho_D_new / rho_new;
            // Calculate new grain radius in cell
            Real da     = da_dt * dt;
            Real a_new  = a + da;
                 a_new *= CMTOMICRON;  // Convert back into microns
                 a_new  = std::max(dust.a_min,a_new);
            // Update scalars
            pmb->pscalars->s(CLOC,k,j,i) = col * rho_new;
            pmb->pscalars->r(ZLOC,k,j,i) = z_new;
            pmb->pscalars->s(ZLOC,k,j,i) = z_new * rho_new;
            pmb->pscalars->r(ALOC,k,j,i) = a_new;
            pmb->pscalars->s(ALOC,k,j,i) = a_new * rho_new;
          }
        }
      }
    }
  }
  // Finished!
  return;
}
//========================================================================================
//! \fn orbitCalc
//! \brief Calculate the position and velocities of the stars based on the model time.
//! Updates orbits, stagnation point position estimation and distances
//! Note: This code is in need of serious optimisation and rewriting, it works, but 
//!       it could be better.
//========================================================================================
void orbitCalc(Real t) {
  // Calculate orbital offset due to time
  Real time_offset = phaseoff * period;  // Adjusted orbital offset
  Real t_orbit     = t + time_offset;    // Time in orbit (s)
  Real phase       = t_orbit / period;   // Orbital phase (fraction)
  // Calculate other orbital properties
  Real phi = 2.0 * PI * phase;
  // Make first guess at eccentric anomaly
  Real E = phi;
  // Calculate trinonometric components
  Real cos_E = std::cos(E);
  Real sin_E = std::sin(E);
  // Use Newton-Raphson solver to calculate E
  Real dE = (phi -E + ecc*sin_E) / (1.0 - ecc*cos_E);
  // Loop to minimise dE
  while (std::abs(dE) > 1e-10) {
    E = E + dE;
    cos_E = std::cos(E);
    sin_E = std::sin(E);
    dE = (phi - E + ecc*sin_E)/(1.0 - ecc*cos_E);
  }

  Real sii   = (std::sqrt(1.0 - ecc*ecc))*sin_E/(1.0 - ecc*cos_E);
  Real coi   = (cos_E - ecc)/(1.0 - ecc*cos_E);
  Real theta = std::atan2(sii,coi);
  
  if (theta < 0.0) {
    theta = 2.0*PI + theta;
  }

  // Calculate radius vector
  Real rrel = 1.0 - ecc*cos_E;
  // Compute barycentric orbital positions and velocities for both stars
  Real m1 = star[WR].mass;
  Real m2 = star[OB].mass;
  // Calculate reduced mass for star 1
  Real RM1 = CUBE(m2) / SQR(m1+m2);
  // Calculate orbital components for star 1
  Real a1 = pow((G*RM1*SQR(period)/(4*SQR(PI))),ONE_3RD);
  Real v1 = sqrt(G*RM1*(2.0/rrel/a1 - 1.0/a1));
  // Calculate reduced mass for star 2
  Real RM2 = CUBE(m1) / SQR(m1+m2);
  // Calculate orbital components for star 2
  Real a2 = pow((G*RM2*SQR(period)/(4*SQR(PI))),ONE_3RD);
  Real v2 = sqrt(G*RM2*(2.0/rrel/a2 - 1.0/a2));

  // Compute angle between velocity and radius vectors
  Real gamma_ang = PI/2.0 + std::acos(std::sqrt((1.0 - SQR(ecc))/(rrel*(2.0 - rrel))));
  Real ang;
  if (theta <= PI) {
    ang = PI - gamma_ang + theta;
  }
  else {
    ang = theta + gamma_ang;
  }

  // Update positions
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real sin_ang   = std::sin(ang);
  Real cos_ang   = std::cos(ang);
  // Since orbit is clockwise on XY plane, SMA along x axis, there is no z orbital component, hence Z component of pos and vel will be 0.0 unless something has gone horribly wrong
  // Star 1
  star[WR].pos[0] = -a1 * rrel * cos_theta;
  star[WR].pos[1] = +a1 * rrel * sin_theta;
  // Star 2
  star[OB].pos[0] = +a2 * rrel * cos_theta;
  star[OB].pos[1] = -a2 * rrel * sin_theta;
  // Update velocities
  // Star 1
  star[WR].vel[0] = -v1 * cos_ang;
  star[WR].vel[1] = +v1 * sin_ang;
  // Star 2
  star[OB].vel[0] = +v2 * cos_ang;
  star[OB].vel[1] = -v2 * sin_ang;
  // Update separation distance
  wcr.updatePositions(star);

  // Update AMR information
  if (amr.enabled) {
    Real desired_resolution = wcr.d_sep / Real(amr.min_cells_between_stars);
    amr.max_level_to_refine = log2(amr.G0dx/desired_resolution);
    if (amr.G0dx/pow(2.0,amr.max_level_to_refine) > desired_resolution) {
      amr.max_level_to_refine++;
    }
    amr.max_level_to_refine += amr.G0_level;
    amr.finest_dx = amr.G0dx;
  }

  // Finished!
  return;
}

//========================================================================================
//! \fn searchAndInterpolate
//! \brief Search through two arrays, X and Y, to find interpolated value for Y given X
//! Used for lookup tables in the context of this problem, two vectors of equal length
//! are required. uses standard C++ algorithm library for binary search, interpolation
//! written for speed, as this function is called millions of times per timestep if
//! cooling is enabled.
Real searchAndInterpolate(std::vector<Real> x_array, std::vector<Real> y_array, Real x) {
  // Perform binary search
  auto upper = std::upper_bound(x_array.begin(),x_array.end(),x);
  // Assign indices based on search
  int iu = std::distance(x_array.begin(),upper);
  int il = iu - 1;
  // Get variables
  Real xl = x_array[il];
  Real xu = x_array[iu];
  Real yl = y_array[il];
  Real yu = y_array[iu];
  // Perform linear interpolation
  Real y = yl + ((x - xl) * ((yu - yl) / (xu - xl)));
  return y;
}