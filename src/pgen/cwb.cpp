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
// Indices for stars
#define WR 0 // Index for Wolf-Rayet, always primary star
#define OB 1 // Index for OB star, always secondary
// Conversions
#define MSOLTOGRAM 1.9884099e+33
#define YEARTOSEC  31556926
#define MSOLYRTOGS 6.3010252e+25
// Physical Constants
// Pi is already defined in defs.hpp!
#define KBOLTZ 1.3806490e-16 // Boltmann constant in CGS (erg/K)
#define RSOL   6.9599000e10  // Solar radius in CGS (cm)
#define MASSH  1.6735575e-24 // Hydrogen mass (g)
#define G      6.6743000e-8  // Gravitational constant in CGS (dyn cm^2/g^2)
// Functions
#define CUBE(x) ( (x)*(x)*(x) )     // Preprocessor cube function, faster than pow(x,3.0)
#define POW4(x) ( (x)*(x)*(x)*(x) ) // Preprocessor x^4 function, faster than pow(x,4.0)
// End of preprocessor definitions

// Functions
int RefinementCondition(MeshBlock *pmb);
void orbitCalc(Real t);
Real searchAndInterpolate();

// Classes
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
    int  ncells_remap = 10;     // Number of cells width for remap radius
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
    void calcWindMomentum(void) {
      mom = mdot * vinf;
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

class WindCollisionRegion {
  public:
    Real d_sep; // Separation distance (cm)
    Real frac_rwr; 
    Real frac_rob;
    Real r_wr;  // Distance from stagnation point to WR star (cm)
    Real r_ob;  // Distance from stagnation point to OB star (cm)
    Real eta;   // Wind momentum ratio
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

class Cooling {
  public:
    bool enabled = false;
};

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

class DustCooling {
  public:
    bool enabled = false;
};

class IonCurve {
  public:
    std::string ion_frac_file_name; // String for filename of ion frac
    // Ionisation fraction curve
    std::vector<Real> logT;  // Ionisation curve log(T) (log(K))
    std::vector<Real> T;     // Ionisation curve temperature (K)
    std::vector<Real> ne;    // Ionisation curve free electrons per ion
    // Dustcooling functions and such can be stored here
    Real T_min = 0.0;
    Real T_max = 0.0;
    int  n_bins;
    // Functions
};

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

// Global variables
// Stars
Star star[NWIND];  // Two stars, WR for index 0 and OB for index 1
Real period;   // Orbital period (s)
Real ecc;      // Orbit eccentricity
Real phaseoff; // Phase offset
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
// Thermodynamics
const Real tmin = 1e4;
Real gmma;  // Gamma
Real gmma1; // Gamma - 1, comes up a lot


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
    cool_curve[WR].curve_file_name = pin->GetString("problem","ccurve1");
    cool_curve[OB].curve_file_name = pin->GetString("problem","ccurve1");
  }
  if (dust.enabled) {
    dust.z_init = pin->GetReal("problem","z_init");
    dust.a_init = pin->GetReal("problem","a_init");
  }
  // Thermodynamic properties
  // gmma  = peos->GetGamma();
  // gmma1 = gmma - 1.0;

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
    EnrollUserRefinementCondition(RefinementCondition);
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

  // Calculate orbit for the first time
  orbitCalc(time);

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
      printf("> Cooling properties\n");
      std::cout << ">  WR Filename: " << cool_curve[WR].curve_file_name << "\n";
      std::cout << ">  OB Filename: " << cool_curve[OB].curve_file_name << "\n";
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
//         then maps on initial density and pressure.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

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
//========================================================================================
void Mesh::UserWorkInLoop() {
  orbitCalc(time+dt);

  // Optional check to see if orbits are moving correctly
  // if (Globals::my_rank == 0) {
  //   printf("t = %.6e\n",time+dt);
  //   for (int n = 0 ; n < 3 ; n++) {
  //     printf("D%d = %.6e %.6e\n",n,star[WR].pos[n],star[OB].pos[n]);
  //   }
  // }

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop(ParameterInput *pin)
//! \brief Check radius of sphere to make sure it is round
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

  // Second run through MeshBlock, limit wind colour
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
        pscalars->s(3,k,j,i) = loc.level * rho;
      }
    }
  }

  // Finished!
  return;
}


//========================================================================================
//! \fn int RefinementCondition(ParameterInput *pmb)
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
// ========================================================================================

int RefinementCondition(MeshBlock *pmb) {
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

// All functions not related to Mesh:: or MeshBlock:: class below this line
// =======================================================================================

// Calculate the position and velocities of the stars based on the model time.
//TODO this code needs some cleaning up 
void orbitCalc(Real t) {
  // Calculate orbital offset due to time
  Real time_offset = phaseoff * period;  // Adjusted orbital offset
  Real t_orbit     = t + time_offset;    // Time in orbit (s)
  Real phase       = t_orbit / period;   // Orbital phase (fraction)
  // Calculate other orbital properties
  Real phi = 2.0 * PI * phase;
  // Make first guess at eccentric anomaly
  Real E = phi;
  // 
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