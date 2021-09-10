//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cwb.cpp
//! \brief Problem generator for 3D Colliding Wind Binary problem with dust advection and
//!        growth, simulation 


// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

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

// Preprocessor definitions
// Indices for scalars, makes for more readable scalar
#define CLOC 0  // Index for wind "colour"
#define ZLOC 1  // Index for dust-to-gas mass ratio, z
#define ALOC 2  // Index for dust grain radius, a
// Indices for stars
#define WR 0 // Index for Wolf-Rayet, always primary star
#define OB 1 // Index for OB star, always secondary
// Conversion values
#define MSOLTOGRAM 1.9884099e+33
#define YEARTOSEC  31556926
#define MSOLYRTOGS 6.3010252e+25
// Constants
// Pi is already defined in defs.hpp!
#define KBOLTZ 1.3806490e-16 // Boltmann constant in CGS (erg/K)
#define RSOL   6.9599000e10  // Solar radius in CGS (cm)
#define MASSH  1.6735575e-24 // Hydrogen mass (g)
#define G      6.6743000e-8  // Gravitational constant in CGS (dyn cm^2/g^2)

// Functions
#define CUBE(x) ( (x)*(x)*(x) )  // Preprocessor cube function, faster than pow(x,3.0)
// End of preprocessor definitions

// Functions
int RefinementCondition(MeshBlock *pmb);

// Classes
class Star {
  public:
    Real mass; // Star mas (g) 
    Real mdot; // Mass loss rate (g/s)
    Real vinf; // Wind terminal velocity (cm/s)
    Real pos[3] = {0.0,0.0,0.0}; // X Y and Z coordinates for star (Cartesian, cm)
    // Wind abundances, stored in an array with followind indexes
    // 0: Hydrogen
    // 1: Helium
    // 2: Carbon
    // 3: Nitrogen
    // 4: Oxygen
    // Without modification, assumes a pure hydrogen flow
    // Assume that all values should come to around 1.0, all other elements are trace
    Real mass_frac[5] = {1.0,0.0,0.0,0.0,0.0}; // 
    Real avg_mass     = MASSH; // Average particle mass (g)
    Real mu           = 1.0;   // Mean molecular mass

    // Functions
    // Calculate average mass, 
    void calcAvgMass(void) {
      const Real m_E[5] = {1.00*MASSH,  // Hydrogen mass
                           4.00*MASSH,  // Helium mass
                           12.0*MASSH,  // Carbon mass
                           14.0*MASSH,  // Nitrogen mass 
                           16.0*MASSH}; // Oxygen mass
      Real avgm = 0.0;
      for (int n = 0; n < 5; n++) {
        avgm += m_E[n] * mass_frac[n];
      }
      avg_mass = avgm;
      mu       = avgm/MASSH;
    }
};

class DustDefaults {
  public:
    bool enabled = false;
    const Real z_min  = 1e-8; // Minimum dust-to-gas mass ratio
    const Real a_min  = 1e-6; // Minimum grain radius (micron)
    Real z_init = z_min; // Initial d2g mass ratio, defined in problem file
    Real a_init = a_min; // Initial grain radius, defined in problem file
};

class Cooling {
  public:
    bool enabled = false;
    std::string curve_file_name;
    // Will be used to store cooling curves
};

class DustCooling {
  public:
    bool enabled = false;
    // Dustcooling functions can be stored here
};

// Global variables
// Stars
Star star[2];  // Two stars, WR for index 0 and OB for index 1
Real eta;      // Wind momentum ratio 
Real period;   // Orbital period (s)
Real ecc;      // Orbit eccentricity
Real phaseoff; // Phase offset 
// Simulation features
bool adaptive     = false; // Does simulation utilise AMR?
// Dust properties
DustDefaults dust; // Initial dust properties
// Cooling objects
Cooling cooling;
DustCooling dust_cooling;

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initialise problem, read in problem file and enroll functions
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read input file
  // WR wind variables
  star[WR].mass = pin->GetReal("problem","mstar1");
  star[WR].mdot = pin->GetReal("problem","mdot1");
  star[WR].vinf = pin->GetReal("problem","vinf1");
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
    cooling.curve_file_name = pin->GetString("problem","ccurve");
  }
  if (dust.enabled) {
    dust.z_init = pin->GetReal("problem","z_init");
    dust.a_init = pin->GetReal("problem","a_init");
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
  // Calculate eta
  eta = (star[OB].mdot * star[OB].vinf) / (star[WR].mdot * star[WR].vinf);
  // Calculate mass fractions for both stars using builtin class function
  star[WR].calcAvgMass();
  star[OB].calcAvgMass();
  // Calculate orbit for the first time
  // calcOrbit();

  // If simulation uses an adaptive mesh refinement code, 
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
  }

  // Setup complete, print out variables
  if (Globals::my_rank ==0) {
    printf("!!! Setup complete!\n");

    printf("> Features\n");
    printf(">  Cooling:      %s\n", cooling.enabled ? "True" : "False");
    printf(">  Dust:         %s\n", dust.enabled ? "True" : "False");
    printf(">  Dust cooling: %s\n", dust_cooling.enabled ? "True" : "False");

    printf("> Star properties\n");
    printf(">  Star masses:  %.3e %.3e g\n",star[WR].mass,star[OB].mass);
    printf(">  Mass loss:    %.3e %.3e g/s\n",star[WR].mdot,star[OB].mdot);
    printf(">  Terminal vel: %.3e %.3e cm/s\n",star[WR].vinf,star[OB].vinf);

    printf("> Orbital properties\n");
    printf(">  Period:   %.3e \n",period);
    printf(">  Phaseoff: %.3f \n",phaseoff);
    printf(">  Ecc:      %.3f \n",ecc);
    printf(">  X:        %.3e %.3e cm\n",star[WR].pos[0],star[OB].pos[0]);
    printf(">  Y:        %.3e %.3e cm\n",star[WR].pos[1],star[OB].pos[1]);
    printf(">  Z:        %.3e %.3e cm\n",star[WR].pos[2],star[OB].pos[2]);
    
    printf("> Abundances\n");
    printf(">  xH:   %.3f %.3f\n",star[WR].mass_frac[0],star[OB].mass_frac[0]);
    printf(">  xHe:  %.3f %.3f\n",star[WR].mass_frac[1],star[OB].mass_frac[1]);
    printf(">  xC:   %.3f %.3f\n",star[WR].mass_frac[2],star[OB].mass_frac[2]);
    printf(">  xN:   %.3f %.3f\n",star[WR].mass_frac[3],star[OB].mass_frac[3]);
    printf(">  xO:   %.3f %.3f\n",star[WR].mass_frac[4],star[OB].mass_frac[4]);
    printf(">  Mu:   %.3f %.3f\n",star[WR].mu,star[OB].mu);
    printf(">  AvgM: %.3e %.3e\n",star[WR].avg_mass,star[OB].avg_mass);

    if (dust.enabled) {
      printf("> Dust properties\n");
      printf(">  a_init: %.3e\n",dust.a_init);
      printf(">  z_init: %.3e\n",dust.z_init);
    }

    if (cooling.enabled) {
      printf("> Cooling properties\n");
      std::cout << ">  Filename: " << cooling.curve_file_name << "\n";
    }
  
    printf("!!! Starting procesing now!\n");
  }
  
  exit(EXIT_SUCCESS);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Colliding Wind Binary problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop(ParameterInput *pin)
//! \brief Check radius of sphere to make sure it is round
//========================================================================================

void MeshBlock::UserWorkInLoop() {

}

//========================================================================================
//! \fn int RefinementCondition(ParameterInput *pmb)
//! \brief Refinement conditions for problem, if AMR is enabled, these will be enabled
//========================================================================================
int RefinementCondition(MeshBlock *pmb) {
  
}
