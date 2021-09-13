//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cwb.cpp
//  \brief Problem generator for CWB problem
//
// The CWB problem consists of two supersonic winds that collide together
// Input parameters are:
//    - problem/mdot1  = mass-loss rate of star1 (Msol/yr)
//    - problem/mdot2  = mass-loss rate of star2 (Msol/yr)
//    - problem/vinf1  = terminal wind speed of star1 (cm/s)
//    - problem/vinf2  = terminal wind speed of star2 (cm/s)
//    - problem/xpos1  = coordinate-1 position of star1 (cm)
//    - problem/ypos1  = coordinate-2 position of star1 (cm)
//    - problem/zpos1  = coordinate-3 position of star1 (cm)
//    - problem/xpos2  = coordinate-1 position of star2 (cm)
//    - problem/ypos2  = coordinate-2 position of star2 (cm)
//    - problem/zpos2  = coordinate-3 position of star2 (cm)
//
// Orbital motion is only included for cartesian calculations (not cylindrically symmetric).
// Wind colour uses the first advected scalar (index 0).
// Dust uses the second and third advected scalars (indices 1 and 2).
// Restrict cooling at unresolved interfaces only applies to cells within the MeshBlock.
// Interfaces that exist between neighbouring MeshBlocks can still have high cooling rates.
//
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"   // JMP: needed for scalars

bool AMR;
bool cool, force, dust;

int minimumNumberOfCellsBetweenStars;
Real G0dx;
Real finestdx;

Real initialDustToGasMassRatio;
Real initialGrainRadiusMicrons;

Real mstar1, mstar2;
Real rstar1, rstar2;
Real teff1, teff2;
Real sigmae1, sigmae2;
Real avgmass1, avgmass2;
Real vtherm1, vtherm2;
Real alpha1, alpha2;
Real k1, k2;

// Array for storing mass fractions, first index is for star, 0 for WR 1 for OB
// Second index is for specific element in the form:
// 0: Hydrogen
// 1: Helium
// 2: Carbon
// 3: Nitrogen
// 4: Oxygen
// This assumes that all other elements are neglibile
Real massFrac[2][5];     // Mass fractions


Real mdot1, mdot2, vinf1, vinf2;
Real eta, fracrob;
Real remapRadius1, remapRadius2; 
Real xpos1, xpos2, ypos1, ypos2, zpos1, zpos2;
Real xvel1, xvel2, yvel1, yvel2, zvel1, zvel2;
Real dsep, rob;
Real stagx, stagy, stagz; // Position of the stagnation point  

Real period;  // orbit period (s)  
Real phaseoff;// phase offset of orbit (from periastron) 
Real ecc;     // orbit eccentricity

Real tmin,tmax; // min/max temperature allowed on grid

int G0_level = 0;
int maxLevelToRefineTo;
int maxLevelForStars;

// Physical and mathematical constants
const Real pi = 2.0*asin(1.0);
const Real Msol = 1.9891e33;
const Real Rsol = 6.9599e10;
const Real yr = 3.15569e7;
const Real boltzman = 1.380658e-16;
const Real massh = 1.67e-24;
const Real stefb = 5.67051e-5;
const Real G = 6.67259e-8;
const Real cspeed = 2.99792458e10;


// User defined constants
const int num_off = 8;
const Real minimumDustToGasMassRatio = 1.0e-10;
const Real minimumGrainRadiusMicrons = 0.001;
const Real grainBulkDensity = 3.0;                       // (g/cm^3)
const Real Twind = 1.0e4;                                // K
const Real avgmass = 1.0e-24;     // g


void PhysicalSources(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Structure to store cooling curve data
struct coolingCurve{
  int ntmax;
  std::string coolCurveFile;
  std::vector<Real> logt,lambdac,te,loglambda;
  Real t_min,t_max,logtmin,logtmax,dlogt;
};

// Structure to store CAK data
struct cakData{
  int ncakmax;
  std::string cakFile;
  std::vector<Real> radius,density,velocity;
  const Real tol=1.0e-3;
  Real mstar,rstar,teff,sigmae,avgmass,vtherm,alpha,k; 
};

const int ncak = 2; // number of CAK data files
cakData cd[ncak];

// JMP prototypes
void AdjustPressureDueToCooling(int is,int ie,int js,int je,int ks,int ke,Real gmma1,AthenaArray<Real> &dei,AthenaArray<Real> &cons);
Real DustCreationRateInWCR(MeshBlock *pmb, int iout);
void EvolveDust(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
Real Interp(Real x, Real xarr[2], Real yarr[2]);
int Hunt(Real* xx, int n, Real x, int jlo);
void OrbitCalc(Real t);
void PhysicalSources(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real Polint(Real* xa, Real* ya, const int n, const Real x);
void RadiateExact(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
void RadiateHeatCool(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
void ReadInCAKdataFiles();
int RefinementCondition(MeshBlock *pmb);
void RestrictCool(int is,int ie,int js,int je,int ks,int ke,int nd,Real gmma1,AthenaArray<Real> &dei,const AthenaArray<Real> &cons);
void StellarForces(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, AthenaArray<Real> &cons);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor. Also called when restarting.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //std::cout << "[Mesh::InitUserMeshData]\n";
  
  mdot1 = pin->GetReal("problem","mdot1");
  mdot2 = pin->GetReal("problem","mdot2");
  vinf1 = pin->GetReal("problem","vinf1");
  vinf2 = pin->GetReal("problem","vinf2");
  // Read in WR wind mass fractions
  massFrac[0][0] = pin->GetReal("problem","xH1");
  massFrac[0][1] = pin->GetReal("problem","xHe1");
  massFrac[0][2] = pin->GetReal("problem","xC1");
  massFrac[0][3] = pin->GetReal("problem","xN1");
  massFrac[0][4] = pin->GetReal("problem","xO1");
  // Read in OB wind mass fractions
  massFrac[1][0] = pin->GetReal("problem","xH2");
  massFrac[1][1] = pin->GetReal("problem","xHe2");
  massFrac[1][2] = pin->GetReal("problem","xC2");
  massFrac[1][3] = pin->GetReal("problem","xN2");
  massFrac[1][4] = pin->GetReal("problem","xO2");  
 
  xpos1 = pin->GetReal("problem","xpos1");
  ypos1 = pin->GetReal("problem","ypos1");
  zpos1 = pin->GetReal("problem","zpos1");
  xpos2 = pin->GetReal("problem","xpos2");
  ypos2 = pin->GetReal("problem","ypos2");
  zpos2 = pin->GetReal("problem","zpos2");

  mstar1 = pin->GetReal("problem","mstar1");
  mstar2 = pin->GetReal("problem","mstar2");

  ecc   = pin->GetReal("problem","ecc");
  period = pin->GetReal("problem","period");
  phaseoff = pin->GetReal("problem","phaseoff");

  std::string cooling = pin->GetString("problem","cooling");
  if      (cooling == "on")  cool = true;
  else if (cooling == "off") cool = false;
  else{
    if (Globals::my_rank == 0) std::cout << "cooling value not recognized: " << cooling << "; Aborting!\n";
    exit(EXIT_SUCCESS);
  }
  if (Globals::my_rank == 0){
    if (cool) std::cout << "Cooling = TRUE\n";
    else      std::cout << "Cooling = FALSE\n";
  }

  std::string sforce = pin->GetString("problem","force");
  if      (sforce == "on")  force = true;
  else if (sforce == "off") force = false;
  else{
    if (Globals::my_rank == 0) std::cout << "force value not recognized: " << sforce << "; Aborting!\n";
    exit(EXIT_SUCCESS);
  }
  if (Globals::my_rank == 0){
    if (force) std::cout << "Force = TRUE\n";
    else       std::cout << "Force = FALSE\n";
  }

  std::string dusty = pin->GetString("problem","dust");
  if      (dusty == "on")  dust = true;
  else if (dusty == "off") dust = false;
  else{
    if (Globals::my_rank == 0) std::cout << "dust value not recognized: " << dust << "; Aborting!\n";
    exit(EXIT_SUCCESS);
  }
  if (dust && NSCALARS < 3){
    // Scalars are: 0 = wind colour
    //              1 = dust to gas mass ratio
    //              2 = dust grain radius (microns)
    if (Globals::my_rank == 0) std::cout << "Not enough scalars for dust modelling. NSCALARS = " << NSCALARS << ". Aborting!\n";
    exit(EXIT_SUCCESS);
  }
  if (Globals::my_rank == 0){
    if (dust) std::cout << "Dust = TRUE, NSCALSRS = " << NSCALARS << "\n";
    else      std::cout << "Dust = FALSE\n";
  }
  if (dust){
    initialDustToGasMassRatio = pin->GetReal("problem","initialDustToGasMassRatio");
    initialGrainRadiusMicrons = pin->GetReal("problem","initialGrainRadiusMicrons");
  }

  // Other initialization tasks/calculations. If this function is called because of a restart,
  // the correct restart time is used so OrbitCalc() can be called.
  mstar1 *= Msol;
  mstar2 *= Msol;
  mdot1 *= Msol/yr;
  mdot2 *= Msol/yr;
  eta = mdot2*vinf2/(mdot1*vinf1); // wind mtm ratio
  fracrob = 1.0 - 1.0/(1.0 + std::sqrt(eta));

  // Determine details about the resolution.
  // No mesh blocks exist at this point. But we can get block info from the athinput file.
  int blocksize_nx1 = pin->GetInteger("meshblock", "nx1");
  int blocksize_nx2 = pin->GetInteger("meshblock", "nx2");
  int blocksize_nx3 = pin->GetInteger("meshblock", "nx3");

  int nxBlocks = mesh_size.nx1/blocksize_nx1;
  int nyBlocks = mesh_size.nx2/blocksize_nx2;
  int nzBlocks = mesh_size.nx3/blocksize_nx3;
  int maxBlocks = std::max(nxBlocks,std::max(nyBlocks,nzBlocks));
  G0_level = std::log2(maxBlocks);
  if (std::pow(2.0,G0_level) < maxBlocks) G0_level++;
  Real x1size = mesh_size.x1max - mesh_size.x1min;
  G0dx = x1size/mesh_size.nx1;
  //std::cout << "x1size = " << x1size << "; G0dx = " << G0dx << "\n";

  //std::cout << "adaptive = " << adaptive << "; multilevel = " << multilevel << "\n";
  //exit(EXIT_SUCCESS);

  //std::string srefinement = pin->GetString("mesh", "refinement");
  //if (srefinement == "adaptive") adaptive = true;
  //else adaptive = false;
  if (adaptive) AMR = true;
  else AMR = false; 

  // Specify the resolution requirement
  minimumNumberOfCellsBetweenStars = 100;

  // Calculate the position and velocity of the stars, dsep, and the number of grid levels needed.
  OrbitCalc(time);

  if (force) ReadInCAKdataFiles();

  // Note: it is only possible to have one source function enrolled by the user.
  // EnrollUserExplicitSourceFunction(PhysicalSources);

  if (adaptive==true)
      EnrollUserRefinementCondition(RefinementCondition);  

  // Add a user-defined global output (see https://github.com/PrincetonUniversity/athena-public-version/wiki/Outputs)
  if (dust){
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, DustCreationRateInWCR, "dmdustdt_WCR");
  }
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the CWB test. 
//         This is not called during a restart, so do not put any important stuff in here.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  //std::cout << "[MeshBlock::ProblemGenerator]\n";
  
  Real gmma  = peos->GetGamma();
  Real gmma1 = gmma - 1.0;

  Real dx = pcoord->dx1v(0); // on this MeshBlock

  // Map on wind 1. It is assumed that the stars are initially along the x-axis (ie. ypos1/2 = zpos1/2 = 0.0).
  Real xCD;  // x-position of the contact discontinuity between the winds
  if (xpos1 < xpos2) xCD = xpos1 + dsep - rob;
  else xCD = xpos1 - dsep + rob;
  
  const int m = 4; // number of points to use in the interpolation
  int indx = 0;

  for (int k=ks; k<=ke; k++) {
    Real zc = pcoord->x3v(k) - zpos1;
    Real zc2 = zc*zc;
    for (int j=js; j<=je; j++) {
      Real yc = pcoord->x2v(j) - ypos1;
      Real yc2 = yc*yc;
      for (int i=is; i<=ie; i++) {
        Real xc = pcoord->x1v(i) - xpos1;
        Real xc2 = xc*xc;
        Real r2 = xc2 + yc2 + zc2;
        Real r = std::sqrt(r2);
        Real xy = std::sqrt(xc2 + yc2);
        Real sinphi = xy/r;
        Real cosphi = zc/r;
        Real costhta = xc/xy;
        Real sinthta = yc/xy;
        Real rho, vel; 
        if (force){
          //std::cout << "rcak0 = " << cd[0].radius[0] << "\n";
          //std::cout << "ncakmax = " << cd[0].ncakmax << "\n";
          //exit(EXIT_SUCCESS);
          if (r <= cd[0].radius[0]){ // inside star
            rho = cd[0].density[0];
            vel = cd[0].velocity[0];
          }
          else if (r > cd[0].radius[cd[0].ncakmax-1]){
            rho = mdot1/(4.0*pi*r2*vinf1);
            vel = vinf1;
          }
          else{
            // Find the index value such that r is between cd[0].radius[indx] and cd[0].radius[indx+1] 
            indx = Hunt(&(cd[0].radius[0]),cd[0].ncakmax,r,indx);
            int n = std::min(std::max(indx-(m-1)/2,0),cd[0].ncakmax-m);
            rho = Polint(&(cd[0].radius[n]),&(cd[0].density[n]),m,r);
            vel = Polint(&(cd[0].radius[n]),&(cd[0].velocity[n]),m,r);
          }
          //std::cout << "r = " << r << "; n = " << n << "; rcak = " << cd[0].radius[n-1] << "\n";
          //exit(EXIT_SUCCESS);  

        }
        else{
          rho = mdot1/(4.0*pi*r2*vinf1);
          vel = vinf1;
        }
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          if (xpos1 < xpos2 && pcoord->x1v(i) < xCD || xpos1 > xpos2 && pcoord->x1v(i) > xCD) {
            Real u1 = vel*sinphi*costhta + xvel1;
            Real u2 = vel*sinphi*sinthta + yvel1;
            //Real u1 = vel*sinphi*costhta; // no orbital velocity for the initial wind map
            //Real u2 = vel*sinphi*sinthta;
            Real u3 = vel*cosphi;
            Real pre = (rho/avgmass)*boltzman*Twind;
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*u1;
            phydro->u(IM2,k,j,i) = rho*u2;
            phydro->u(IM3,k,j,i) = rho*u3;
            phydro->u(IEN,k,j,i) = pre/gmma1 + 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
            if (NSCALARS > 0) {
              // pscalars->s(0,k,j,i) = rho;
              pscalars->s(0,k,j,i) = 0.0;
            }
            if (dust){
              // pscalars->s(1,k,j,i) = initialDustToGasMassRatio*rho;
              // pscalars->s(2,k,j,i) = initialGrainRadiusMicrons*rho;
              pscalars->s(1,k,j,i) = 0.0;
              pscalars->s(2,k,j,i) = 0.0;
            }
          }
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) { // RPZ
          if (pcoord->x3v(k) < xCD) { // x = R; y = P; z = Z
            Real rho = mdot1/(4.0*pi*r2*vinf1);
            Real u1 = vinf1*sinphi;
            Real u2 = 0.0;
            Real u3 = vinf1*cosphi;
            Real pre = (rho/avgmass)*boltzman*Twind;
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*u1;
            phydro->u(IM2,k,j,i) = rho*u2;
            phydro->u(IM3,k,j,i) = rho*u3;
            phydro->u(IEN,k,j,i) = pre/gmma1 + 0.5*rho*vinf1*vinf1;
            if (NSCALARS > 0) {
              pscalars->s(0,k,j,i) = 0.0;
            }
            if (dust){
              pscalars->s(1,k,j,i) = initialDustToGasMassRatio*rho;
              pscalars->s(2,k,j,i) = initialGrainRadiusMicrons*rho;
            }
          }	  
        }
      }
    }
  }

  // Map on wind 2
  for (int k=ks; k<=ke; k++) {
    Real zc = pcoord->x3v(k) - zpos2;
    Real zc2 = zc*zc;
    for (int j=js; j<=je; j++) {
      Real yc = pcoord->x2v(j) - ypos2;
      Real yc2 = yc*yc;
      for (int i=is; i<=ie; i++) {
        Real xc = pcoord->x1v(i) - xpos2;
        Real xc2 = xc*xc;
        Real r2 = xc2 + yc2 + zc2;
        Real r = std::sqrt(r2);
        Real xy = std::sqrt(xc2 + yc2);
        Real sinphi = xy/r;
        Real cosphi = zc/r;
        Real costhta = xc/xy;
        Real sinthta = yc/xy;
        Real rho, vel; 
        if (force){
          if (r <= cd[1].radius[0]){ // inside star
            rho = cd[1].density[0];
            vel = cd[1].velocity[0];
          }
          else if (r > cd[1].radius[cd[1].ncakmax-1]){
            rho = mdot2/(4.0*pi*r2*vinf2);
            vel = vinf2;
          }
          else{
            // Find the index value such that r is between cd[0].radius[indx] and cd[0].radius[indx+1] 
            indx = Hunt(&(cd[1].radius[0]),cd[1].ncakmax,r,indx);
            int n = std::min(std::max(indx-(m-1)/2,0),cd[1].ncakmax-m);
            rho = Polint(&(cd[1].radius[n]),&(cd[1].density[n]),m,r);
            vel = Polint(&(cd[1].radius[n]),&(cd[1].velocity[n]),m,r);
          }
        }
        else{
          rho = mdot2/(4.0*pi*r2*vinf2);
          vel = vinf2;
        }
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          if (xpos1 < xpos2 && pcoord->x1v(i) >= xCD || xpos1 > xpos2 && pcoord->x1v(i) < xCD) {
            Real u1 = vel*sinphi*costhta + xvel2;
            Real u2 = vel*sinphi*sinthta + yvel2;
            //Real u1 = vel*sinphi*costhta; // no orbital velocities for the initial wind map
            //Real u2 = vel*sinphi*sinthta;
            Real u3 = vel*cosphi;
            Real pre = (rho/avgmass)*boltzman*Twind;
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*u1;
            phydro->u(IM2,k,j,i) = rho*u2;
            phydro->u(IM3,k,j,i) = rho*u3;
            phydro->u(IEN,k,j,i) = pre/gmma1 + 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
            if (NSCALARS > 0) {
              pscalars->s(0,k,j,i) = 0.0;
            }
            if (dust){
              // pscalars->s(1,k,j,i) = initialDustToGasMassRatio*rho;
              // pscalars->s(2,k,j,i) = initialGrainRadiusMicrons*rho;
              pscalars->s(1,k,j,i) = 0.0;
              pscalars->s(2,k,j,i) = 0.0;
            }
          }
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) { // RPZ
          if (pcoord->x3v(k) >= xCD) { // x = R; y = P; z = Z
            Real rho = mdot2/(4.0*pi*r2*vinf2);
            Real u1 = vinf2*sinphi;
            Real u2 = 0.0;
            Real u3 = vinf2*cosphi;
            Real pre = (rho/avgmass)*boltzman*Twind;
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*u1;
            phydro->u(IM2,k,j,i) = rho*u2;
            phydro->u(IM3,k,j,i) = rho*u3;
            phydro->u(IEN,k,j,i) = pre/gmma1 + 0.5*rho*vinf2*vinf2;	    
            if (NSCALARS > 0) {
              pscalars->s(0,k,j,i) = 0.0;
            }
            if (dust){
              pscalars->s(1,k,j,i) = initialDustToGasMassRatio*rho;
              pscalars->s(2,k,j,i) = initialGrainRadiusMicrons*rho;
            }
          }	  
        }
	

      }
    }
  }
  
  return;
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
// JMP: This is called after the end of all of the hydro steps (i.e. just before the simulation exits)
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // Cycle over all MeshBlocks
  //MeshBlock *pmb = pblock;
  //while (pmb != nullptr) {
  //  pmb = pmb->next;
  //}
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once after every time step for user-defined work.
//         Because time has not yet been updated we need to add on dt.
//         This is used to update the stellar positions, dsep, etc. 
//         max_level is the maximum level that the simulation could go to, NOT the current maximum level
//========================================================================================
void Mesh::UserWorkInLoop() {
  //if (Globals::my_rank == 0){
  //  std::cout << "root_level = " << root_level << "; max_level = " << max_level << "; current_level = " << current_level << "\n";
  //  exit(EXIT_SUCCESS);
  //}
  OrbitCalc(time+dt);
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//         JMP: I belive this function is called immediately AFTER the hydro work has been done.
//              It is called after calls to DoTaskListOneStage() in step 8 of the main integration
//              loop in main(). The Mesh is constructed in step 4, using either a constructor
//              that takes the athinput file as an argument, or a constructor that takes a restart
//              file as an argument. The mesh is then initialized in step 6. 
//========================================================================================
void MeshBlock::UserWorkInLoop() {

  Real gmma  = peos->GetGamma();
  Real gmma1 = gmma - 1.0;

  Real dt = pmy_mesh->dt; // all MeshBlocks advance the same timestep, on all levels
  
  Real xpos[2]={xpos1,xpos2};
  Real ypos[2]={ypos1,ypos2};
  Real zpos[2]={zpos1,zpos2};
  Real xvel[2]={xvel1,xvel2};
  Real yvel[2]={yvel1,yvel2};
  Real mdot[2]={mdot1,mdot2};
  Real vinf[2]={vinf1,vinf2};
  Real scalar[2]={1.0,0.0};
  
  Real dx = pcoord->dx1v(0); // on this MeshBlock
  Real dy = pcoord->dx1v(0); // on this MeshBlock
  Real dz = pcoord->dx1v(0); // on this MeshBlock

  // It is the users responsibility to ensure that there is always enough resolution to
  // prevent the remapRadius interfering with the WCR and to provide enough resolution
  // when the winds are radiatively driven.
  if (force){
    remapRadius1 = cd[0].rstar + 3.0*finestdx; // +3dx
    remapRadius2 = cd[1].rstar + 3.0*finestdx;
    //std::cout << "remapRadius1 = " << remapRadius1 << "; rstar1 = " << cd[0].rstar << "; finestdx = " << finestdx << "\n";
    //exit(EXIT_SUCCESS);
  }
  else{
    remapRadius1 = 6.0*dx;
    remapRadius2 = 6.0*dx; 
  }
  Real remapRadius[2] = {remapRadius1,remapRadius2};  
  int remapi[2] = {int(remapRadius1/dx),int(remapRadius2/dx)};
  //std::cout << "remapRadius1 = " << remapRadius1 << "; remapi1 = " << remapi[0] << "; remapRadius2 = " << remapRadius2 << "; remapi2 = " << remapi[1] << "\n";
  //exit(EXIT_SUCCESS);

  int indx = 0;
  const int m = 4; // number of points to use in interpolation

  // Remap winds
  for (int nw = 0; nw < 2; ++nw){ // Loop over each wind
    int istar = int((xpos[nw] - pcoord->x1f(0))/pcoord->dx1f(0));
    int jstar = int((ypos[nw] - pcoord->x2f(0))/pcoord->dx2f(0));
    int kstar = int((zpos[nw] - pcoord->x3f(0))/pcoord->dx3f(0));
    int istl = std::max(is,istar-remapi[nw]-2);
    int jstl = std::max(js,jstar-remapi[nw]-2);
    int kstl = std::max(ks,kstar-remapi[nw]-2);
    int istu = std::min(ie,istar+remapi[nw]+2);
    int jstu = std::min(je,jstar+remapi[nw]+2);
    int kstu = std::min(ke,kstar+remapi[nw]+2);
    if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
      jstl = js; jstu = je;
    }
    for (int k=kstl; k<=kstu; k++) {
      Real zc = pcoord->x3v(k) - zpos[nw]; // volume centered I think
      Real zc2 = zc*zc;
      for (int j=jstl; j<=jstu; j++) {
        Real yc = pcoord->x2v(j) - ypos[nw];
        Real yc2 = yc*yc;
        for (int i=istl; i<=istu; i++) {
          Real xc = pcoord->x1v(i) - xpos[nw];
	        Real xc2 = xc*xc;
  	      Real r2 = xc2 + yc2 + zc2;
	        Real r = std::sqrt(r2);
          if (r < remapRadius[nw]) {
            // Loop over the volume of the cell to calculate volume weighted averages
            // dx1f, dx2f, dx3f, x1f, x2f, x3f are face spacing and positions
            // dx1v, dx2v, dx3v, x1v, x2v, x3v are volume spacing and positions
            //const int nimax = 5; const int njmax = 5; const int nkmax = 5;
            const int nimax = 1; const int njmax = 1; const int nkmax = 1;
            Real dnx = dx/Real(nimax);
            Real dny = dy/Real(njmax);
            Real dnz = dz/Real(nkmax);

            Real rhotot = 0.0; Real mtm1 = 0.0; Real mtm2 = 0.0; Real mtm3 = 0.0;

            for (int nk = 0; nk < nkmax; nk++){
              Real z = pcoord->x3f(k) + (Real(nk)+0.5)*dnz - zpos[nw];
              for (int nj = 0; nj < njmax; nj++){
                Real y = pcoord->x2f(j) + (Real(nj)+0.5)*dny - ypos[nw];
                for (int ni = 0; ni < nimax; ni++){
                  Real x = pcoord->x1f(i) + (Real(ni)+0.5)*dnx - xpos[nw];
                  Real r = std::sqrt(x*x + y*y + z*z);

                  Real xy = std::sqrt(x*x + y*y);
                  Real sinphi = xy/r;
                  Real cosphi = z/r;
                  Real costhta = x/xy;
                  Real sinthta = y/xy;
            
                  Real rho,vel;
                  if (force){
                    if (r <= cd[nw].radius[0]){ // inside star
                      rho = cd[nw].density[0];
                      vel = cd[nw].velocity[0];
                    }
                    else if (r > cd[nw].radius[cd[nw].ncakmax-1]){
                      rho = mdot[nw]/(4.0*pi*r2*vinf[nw]);
                      vel = vinf[nw];
                    }
                    else{
                      // Find the index value such that r is between cd[nw].radius[indx] and cd[nw].radius[indx+1] 
                      indx = Hunt(&(cd[nw].radius[0]),cd[nw].ncakmax,r,indx);
                      int n = std::min(std::max(indx-(m-1)/2,0),cd[nw].ncakmax-m);
                      rho = Polint(&(cd[nw].radius[n]),&(cd[nw].density[n]),m,r);
                      vel = Polint(&(cd[nw].radius[n]),&(cd[nw].velocity[n]),m,r);
                    }
                  }
                  else{
                    rho = mdot[nw]/(4.0*pi*r2*vinf[nw]);
                    vel = vinf[nw];
                  }
                  Real u1 = vel*sinphi*costhta + xvel[nw];
                  Real u2 = vel*sinphi*sinthta + yvel[nw];
                  Real u3 = vel*cosphi;

                  // Sum up
                  rhotot += rho;
                  mtm1 += rho*u1;
                  mtm2 += rho*u2;
                  mtm3 += rho*u3;
                }
              }
            }
            int nsamples = nimax*njmax*nkmax;
            Real rho = rhotot/Real(nsamples);
            mtm1 /= Real(nsamples);
            mtm2 /= Real(nsamples);
            mtm3 /= Real(nsamples);
            Real u1 = mtm1/rho;
            Real u2 = mtm2/rho;
            Real u3 = mtm3/rho;
            Real pre = (rho/avgmass)*boltzman*Twind;

            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*u1;
            phydro->u(IM2,k,j,i) = rho*u2;
            phydro->u(IM3,k,j,i) = rho*u3;
            phydro->u(IEN,k,j,i) = pre/gmma1 + 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
	    
            // Set passive scalars
            if (NSCALARS > 0) {
              // wind "colour"
              // pscalars->s(0,k,j,i) = 0.0;
              // pscalars->r(0,k,j,i) = scalar[nw];
              pscalars->s(0,k,j,i) = scalar[nw]*rho;
            }
      	    if (dust){
              pscalars->s(1,k,j,i) = initialDustToGasMassRatio*rho;
              pscalars->s(2,k,j,i) = initialGrainRadiusMicrons*rho;
              // pscalars->r(1,k,j,i) = initialDustToGasMassRatio;
              // pscalars->r(2,k,j,i) = initialGrainRadiusMicrons;
	          }
          }
        }
      }
    }
  }


  // Limit wind colour
  if (NSCALARS > 0) {  
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
	  // wind "colour"
          // pscalars->s(0,k,j,i) = std::min(std::max(pscalars->s(0,k,j,i),0.0),1.0);
          pscalars->s(0,k,j,i) = std::min(std::max(pscalars->s(0,k,j,i),0.0),phydro->u(IDN,k,j,i));
          // Real rho = phydro->u(IDN,k,j,i);
          // Real col = pscalars->s(0,k,j,i);
          //      if (col > rho) col = rho;
          // else if (col < 0.0) col = 0.0;
          // pscalars->s(0,k,j,i) = col;
        }
      }
    }
  }

  return;
}

int Hunt(Real* xx, int n, Real x, int jlo){
  // Given an array xx and given a value x, return a value jlo such that x is
  // inbetween xx[jlo] and xx[jlo+1]. xx must be monotonic, but may be either increasing
  // or decreasing. jlo = -1 or jlo = n-1 is returned to indicate that x is out of range.
  // jlo on input is taken as the initial guess of jlo on output.
  // JMP 25/03/21 - Based on search.c

  int inc = 1; // set the hunting increment
  int jhi;
  bool ascnd = (xx[n-1] >= xx[0]);
  if (jlo < 0 || jlo > n-1){
    // Input guess not useful. Go immediately to bisection.
    jlo = -1;
    jhi = n;
    goto bisection;
  }
  if ((x >= xx[jlo]) == ascnd){ // hunt up
  part1:;
    jhi = jlo + inc;
    if (jhi > n-1){ // done hunting since off end of table
      jhi = n;
    }
    else if ((x >= xx[jhi]) == ascnd){ // not done hunting
      jlo = jhi;
      inc += inc; // double the increment
      goto part1; // and try again
    } // done hunting - value bracketed
  }
  else{ // hunt down
    jhi = jlo;
  part2:;
    jlo = jhi - inc;
    if (jlo < 0){ // done hunting since off end of table
      jlo = -1;
    }
    else if ((x < xx[jlo]) == ascnd){ // not done hunting
      jhi = jlo;
      inc += inc; // double the increment
      goto part2; // and try again
    } // done hunting - value bracketed
  } 
  // The hunt is done. Now begin the final bisection phase.
 bisection:;
  while (true){
    if (jhi - jlo == 1){
      if (x == xx[n-1]) jlo = n-2;
      if (x == xx[0])   jlo = 0;
      return jlo;
    }
    int jm = (jhi+jlo)/2;
    if ((x >= xx[jm]) == ascnd){
      jlo = jm;
    }
    else{
      jhi = jm;
    }
  }  
}

// Given arrays xa[0..n-1] and ya[0..n-1], and given a value x, this routine
// returns the interpolated value of y.
// JMP 06/11/19 - Correctly working.
Real Polint(Real* xa, Real* ya, const int n, const Real x){
  Real result = 0.0;
  for (int i = 0; i < n; i++){ 
    // Compute individual terms of Lagrange's interpolation formula
    Real term = ya[i];
    for (int j = 0; j < n; j++){ 
      if (j != i){ 
        term *= (x - xa[j])/(xa[i] - xa[j]); 
      } 
    }
    // Add current term to result 
    result += term; 
  }
  return result;
}

// Linearly interpolate between two points 
Real Interp(Real x, Real xarr[2], Real yarr[2]){
  Real grad = (yarr[1] - yarr[0])/(xarr[1] - xarr[0]);
  Real y = yarr[0] + grad*(x-xarr[0]);
  return y;
}

// Refinement condition. We can use velocity convergence to find the central parts of the WCR.
// In 2D RPZ geometry, i(1) is R, j(2) is P and k(3) is Z, and the stars are located along Z (k).
// Therefore there should be a velocity convergence in Z at the stagnation point.
int RefinementCondition(MeshBlock *pmb){

  AthenaArray<Real> &w = pmb->phydro->w;
  // AthenaArray<Real> &r = pmb->pscalars->r;

  // Mesh contains: root_level, max_level, current_level;
  // The Mesh starts with a single MeshBlock on level 0. Thus the G0 blocks are on the
  // G0_level. Determine this level.
  // NOTE: The following calculation cannot be done in Mesh::InitUserMeshData() because
  // no MeshBlocks exist at that time.

/*
// This is now done in Mesh::InitUserMeshData and at the end of OrbitCalc(). It might be that this won't work properly
// with restarts, in which case it may be necessary to uncomment the below...
  int nxBlocks = pmb->pmy_mesh->mesh_size.nx1/pmb->block_size.nx1;
  int nyBlocks = pmb->pmy_mesh->mesh_size.nx2/pmb->block_size.nx2;
  int nzBlocks = pmb->pmy_mesh->mesh_size.nx3/pmb->block_size.nx3;
  int maxBlocks = std::max(nxBlocks,std::max(nyBlocks,nzBlocks));
  G0_level = std::log2(maxBlocks);
  if (std::pow(2.0,G0_level) < maxBlocks) G0_level++;

  // Specify and determine the resolution requirements
  int minimumNumberOfCellsBetweenStars = 100;
  Real x1size = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  Real G0dx = x1size/pmb->pmy_mesh->mesh_size.nx1;
  //std::cout << "x1size = " << x1size << "; G0dx = " << G0dx << "\n";
  Real desiredResolution = dsep/Real(minimumNumberOfCellsBetweenStars);
  int maxLevelToRefineTo = std::log2(G0dx/desiredResolution);
  if (G0dx/std::pow(2.0,maxLevelToRefineTo) > desiredResolution) maxLevelToRefineTo++;
  maxLevelToRefineTo += G0_level;
*/

  // If the current MeshBlock is on a higher level than the current maximum, immediately derefine.
  // This happens when the separation between the stars is increasing.
  if (pmb->loc.level > maxLevelToRefineTo) return -1; // derefine


  // Now refine on some user specified criteria such as divV < 0.0 (indicating a shock or parts of the WCR)  
  for(int k=pmb->ks-1; k<=pmb->ke+1; k++) {
    Real dz = (pmb->pcoord->x3v(k+1) - pmb->pcoord->x3v(k-1));
    for(int j=pmb->js-1; j<=pmb->je+1; j++) {
      Real dy = (pmb->pcoord->x2v(j+1) - pmb->pcoord->x2v(j-1));
      for(int i=pmb->is-1; i<=pmb->ie+1; i++) {
        Real dx = (pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i-1));
        //std::cout << "dx = " << dx << "; dy = " << dy << "; dz = " << dz << "\n";
        //exit(EXIT_SUCCESS);

        /*
      	// Refine on overdensity
	      Real rho = w(IDN,k,j,i);
        if (NSCALARS > 0){
	        if (r(0,k,j,i) > 0.5){ // In wind 1
            Real zc = pmb->pcoord->x3v(k) - zpos1;
            Real yc = pmb->pcoord->x2v(j) - ypos1;
            Real xc = pmb->pcoord->x1v(i) - xpos1;
    	      Real r2 = xc*xc + yc*yc + zc*zc;
	          Real rhoWind = mdot1/(4.0*pi*r2*vinf1);
	          if (rhoWind > 2.0*rho) return 1; // refine
	        }
	        else{
            Real zc = pmb->pcoord->x3v(k) - zpos2;
            Real yc = pmb->pcoord->x2v(j) - ypos2;
            Real xc = pmb->pcoord->x1v(i) - xpos2;
    	      Real r2 = xc*xc + yc*yc + zc*zc;
	          Real rhoWind = mdot2/(4.0*pi*r2*vinf2);
	          if (rhoWind > 2.0*rho) return 1; // refine
	        }
	     }
       */

	     // Refine on divergence condition (cartesian)
	     Real dudx = (w(IVX,k,j,i+1)-w(IVX,k,j,i-1))/dx;
	     Real dvdy = (w(IVY,k,j+1,i)-w(IVY,k,j-1,i))/dy;
	     Real dwdz = (w(IVZ,k+1,j,i)-w(IVZ,k-1,j,i))/dz;
	     Real divV = dudx + dvdy + dwdz;
	    
       //std::cout << "dudx = " << dudx << "; dvdy = " << dvdy << "; dwdz = " << dwdz << "; divV = " << divV << "\n";
       //exit(EXIT_SUCCESS);
 	     if (divV < 0.0){ // potentially refine
	       // Check to see if not too far away from the stagnation point.
	       Real x = pmb->pcoord->x1v(i) - stagx;
	       Real y = pmb->pcoord->x2v(j) - stagy;
	       Real z = pmb->pcoord->x3v(k) - stagz;
	       Real r = std::sqrt(x*x + y*y + z*z);  // distance from stagnation point
	       int ri = int(r/dx);

         // Based on the number of cells to the stagnation point, either refine, do nothing, or derefine
    	   if (ri < 10){
	         if (maxLevelToRefineTo - pmb->loc.level > 0) return 1; // refine (10)
	       }
	       else if (ri < 30);     // do nothing // 30
	       else return -1;	 // derefine 
	     }

      }
    }
  }

  // Set refinement based on distance to each star
  Real xpos[2]={xpos1,xpos2};
  Real ypos[2]={ypos1,ypos2};
  Real zpos[2]={zpos1,zpos2};

  bool derefineWind = true;
  
  for (int nw = 0; nw < 2; ++nw){ // Loop over each wind
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real zc = pmb->pcoord->x3v(k) - zpos[nw];
      Real zc2 = zc*zc;
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real yc = pmb->pcoord->x2v(j) - ypos[nw];
        Real yc2 = yc*yc;
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real dx = (pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i-1));
          Real xc = pmb->pcoord->x1v(i) - xpos[nw];
	        Real xc2 = xc*xc;
	        Real r2 = xc2 + yc2 + zc2;
	        Real r = std::sqrt(r2);
	        int ri = int(r/dx);

          // Refine, do nothing, or potentially derefine, based on distance from either star
          if (nw == 0){
	          // Refine only to maxLevel-2 near star 1
	          if (maxLevelToRefineTo - pmb->loc.level <= 2){
	            // Do nothing
	          }
	          else if (ri < 10) return 1; // refine (10)
	        }
	        else if (maxLevelToRefineTo - pmb->loc.level > 0 && ri < 10) return 1; // refine (10) for wind 2
	  
	        if (ri < 20) derefineWind = false; // 20
	      }
      }
    }
  }
  
  if (derefineWind) return -1; // derefine
  return 0; // keep as is 
}


// Calculate the position and velocities of the stars based on the model time.
void OrbitCalc(Real t){

  double xdist,ydist,zdist;
  double time_offset,torbit,phase,phi,E,dE,cosE,sinE;
  double sii,coi,theta,ang;
  double rrel;      // radius vector
  double gamma_ang; // angle between velocity and radius vectors
  double a1,a2;     // semi-major axis of barycentric orbits
  double M;         // effective orbit mass
  double m1,m2,v1,v2;

  //std::cout << "dsep = " << dsep << "; t = " << t << "; phaseoff = " << phaseoff << "\n";
  
  time_offset = phaseoff*period;
  torbit = t + time_offset;  // time in seconds
  phase = torbit/period;

  //      write(*,*) T,torbit,phase
  //      quit()

  phi = 2.0*pi*phase;
  E = phi;           // first guess at eccentric anomaly

  // Now use Newton-Raphson solver to calculate E
  // (dsin, dcos, and dabs are double precision versions)

  cosE = std::cos(E);
  sinE = std::sin(E);
  
  dE = (phi - E + ecc*sinE)/(1.0 - ecc*cosE);
 iterate:
  if (std::abs(dE) > 1.0e-10){
    E = E + dE;
    cosE = std::cos(E);
    sinE = std::sin(E);
    dE = (phi - E + ecc*sinE)/(1.0 - ecc*cosE);
    goto iterate;
  }

  sii = (std::sqrt(1.0 - ecc*ecc))*sinE/(1.0 - ecc*cosE);
  coi = (cosE - ecc)/(1.0 - ecc*cosE);
  theta = std::atan2(sii,coi);
  if (theta < 0.0) theta= 2.0*pi + theta;   // 0 < theta < 2pi
         
  rrel = 1.0 - ecc*cosE;   // radius vector

  // Compute barycentric orbital positions and velocities of masses m1 and m2

  m1 = mstar1/Msol;
  m2 = mstar2/Msol;

  double G = 6.67259e-8;

  M = (std::pow(m2,3)/std::pow((m1+m2),2)) * Msol;
  a1 = std::pow((G*M*period*period/(4.0*pi*pi)),(1.0/3.0));
  v1 = std::sqrt(G*M*(2.0/rrel/a1 - 1.0/a1));
  M = (std::pow(m1,3)/std::pow((m1+m2),2)) * Msol;
  a2 = std::pow((G*M*period*period/(4.0*pi*pi)),(1.0/3.0));
  v2 = std::sqrt(G*M*(2.0/rrel/a2 - 1.0/a2));

  // Compute angle between velocity and radius vectors

  gamma_ang = pi/2.0 + std::acos(std::sqrt((1.0 - ecc*ecc)/(rrel*(2.0 - rrel))));

  if (theta <= pi) ang = pi-gamma_ang+theta;
  else             ang = theta+gamma_ang;

  double sintheta = std::sin(theta);
  double costheta = std::cos(theta);
  double sinang = std::sin(ang);
  double cosang = std::cos(ang);

  // Initialise all velocities as these are not provided by the input file
  //xvel1 = 0.0; xvel2 = 0.0; yvel1 = 0.0; yvel2 = 0.0; zvel1 = 0.0; zvel2 = 0.0;

  // Orbit is clockwise in the xy plane, with the semi-major axis along the x-axis
  zpos1 = 0.0;
  zpos2 = 0.0;
  zvel1 = 0.0;
  zvel2 = 0.0;
  ypos1 =  a1*rrel*sintheta;
  xpos1 = -a1*rrel*costheta;
  ypos2 = -a2*rrel*sintheta;
  xpos2 =  a2*rrel*costheta;
  yvel1 =  v1*sinang;
  xvel1 = -v1*cosang;
  yvel2 = -v2*sinang;
  xvel2 =  v2*cosang;

  // Update dsep and rob
  xdist = xpos1 - xpos2;
  ydist = ypos1 - ypos2;
  zdist = zpos1 - zpos2;
  dsep = std::sqrt(xdist*xdist + ydist*ydist + zdist*zdist);
  rob = fracrob*dsep;     //distance of stagnation point from star 1 (distance from star 0 is rwr)
  
  // Update the position of the stagnation point
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    // This works for all x/ypos1/2. i.e. whether x/ypos2 > x/ypos1, or vice-versa 
    stagx = xpos2 - fracrob*(xpos2 - xpos1);
    stagy = ypos2 - fracrob*(ypos2 - ypos1);
    stagz = 0.0;
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    stagx = 0.0;
    stagy = 0.0;
    stagz = zpos2 - rob;
  }

  Real desiredResolution = dsep/Real(minimumNumberOfCellsBetweenStars);
  maxLevelToRefineTo = std::log2(G0dx/desiredResolution);
  if (G0dx/std::pow(2.0,maxLevelToRefineTo) > desiredResolution) maxLevelToRefineTo++;
  maxLevelToRefineTo += G0_level;

  if (AMR){
    finestdx = G0dx/std::pow(2.0,maxLevelToRefineTo - G0_level);
  }
  else finestdx = G0dx;

  //std::cout << "xpos1 = " << xpos1 << "; xpos2 = " << xpos2 << "; ypos1 = " << ypos1 << "; ypos2 = " << ypos2 << "; zpos1 = " << zpos1 << "; zpos2 = " << zpos2 << "\n";
  //std::cout << "dsep = " << dsep << "\n";
  //std::cout << "G0_level = " << G0_level << "\n";
  //std::cout << "G0dx = " << G0dx << "\n";
  //std::cout << "maxLevelToRefineTo = " << maxLevelToRefineTo << "\n";
  //std::cout << "finestdx = " << finestdx << "\n";
  //exit(EXIT_SUCCESS);
  
  return;
}


//========================================================================================
//! \fn void PhysicalSources(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
//		const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
//  \brief Physical source terms
//
//  Within the source function, the conservative variable (cons) should be updated. 
//  If passive scalars are enabled, the species densities (cons_scalar) should also be updated. 
//  This source function is called after the MHD and passive scalar updates but before conservative 
//  to primitive conversion. Therefore, the updates are already reflected in the conservative variables 
//  but the primitive variables (prim and prim_scalar) and cell-centered magnetic fields are not updated 
//  yet. The conservative variables should be updated using these primitive variables and cell-centered 
//  magnetic fields.
//========================================================================================
void PhysicalSources(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
		const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){
  // if (cool) RadiateExact(pmb,dt,cons);
  // if (cool) RadiateHeatCool(pmb,dt,cons);
  // if (force) StellarForces(pmb,dt,prim,cons);
  // if (dust) EvolveDust(pmb,dt,cons);
  // if (dust) EvolveDustMultiWind(pmb,dt,cons);
  return;
}


/*!  \brief Perform an EXACT calculation of radiative cooling using a single cooling curve.
 *
 *   No heating or other form of cooling (e.g. dust emission) can occur.
 * 
 *   For a given cooling curve, calculate the temporal evolution function (TEF)
 *   so that an exact integration scheme for the cooling can be used (see
 *   Townsend 2009, ApJS, 181, 391). 
 *   Based on exactcooling_good.f in other_programs/ExactCooling.
 *   Copied from heatCool.cpp in ~/work/wbb/mtm_injection_2019/arwenFromBSc2018-19
 *
 *   Various possibilities for bugs exist. Current limitations include:
 *    1) The mininum (floor) temperature must not be below the minimum
 *       temperature in the cooling curve.
 *    2) The temperature scales of each cooling curve must be identical.
 *    3) At least 2 cooling curves are expected (these can be identical
 *       if required).
 *    4) The cooling curves must have a uniform logarithmic spacing in 
 *       temperature (because of the guess for the index kk).
 * 
 *   \author Julian Pittard (Original version 13.09.11)
 *   \version 1.0-stable (Evenstar):
 *   \date Last modified: 13.09.11 (JMP)
 */
void RadiateExact(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons){

  const int ncmax = 2;          // maximum number of cooling curves
  const int ntmax = 121;        // maximum number of temperatures in a cooling curve
  //string coolcurve[ncmax];         // cooling curve filenames
  
  bool restrictUnresolvedCooling = true;
  int ncool = 2;                   // actual number of cooling curves (CWB,WBB,SNR = 2)
  
  Real logtemp,y_new,y_temp;
  Real lambda_init[ncmax],temp_new_town;

  // The following variables need their values "preserved" between calls
  // (or alternatively they could be made global or file-scope)
  static bool first = true;
  static Real t_min,t_max,t_ref,logtmin,logtmax,dlogt;
  static Real te[ntmax];                // cooling curve temperatures     (assumed identical between curves)
  static Real logt[ntmax];              // cooling curve log temperatures (assumed identical between curves)
  static Real lambda[ncmax][ntmax];     // cooling curve lambda's
  static Real loglambda[ncmax][ntmax];  // cooling curve log lambda's
  static Real y[ncmax][ntmax];          // cooling curve y's
  static Real alpha[ncmax][ntmax];      // cooling curve alpha's
  static Real lambda_ref[ncmax];        // cooling curve reference lambda's
  static Real fac[ncmax];               // 

  Real gmma  = pmb->peos->GetGamma();
  Real gmma1 = gmma - 1.0;

  tmin = Twind;
  tmax = 3.0e9;

  // If first call to this subroutine we need to read data files from disk.
  if (first){
    // Read in cooling curve(s). It is assumed that the temperature binning is 
    // uniform in log space i.e. dlog(T) is a constant 

    for (int nc = 0; nc < ncool; nc++){
      //ifstream file(coolcurve[nc].c_str()); //C++ method (coolcurve must be a string type)
      //ifstream file("solar0.005-100keV");
      std::ifstream file("cooling_curve_solar_lambdaJMP_fromWang2014.txt"); 
      if (!file) {
        std::cerr << "RadiateExact: failed to open file\n";
        exit(EXIT_FAILURE);
      } 
      //std::cout << "Reading in cooling curve " << coolcurve[nc] << "\n"; 
      for (int n = 0; n < ntmax; n++){
        //std::cout << "n = " << n << "\n";
        file >> logt[n] >> lambda[nc][n];
        te[n] = std::pow(10,logt[n]);
        loglambda[nc][n] = log10(lambda[nc][n]);
        //std::cout << n << "\t" << te[n] << "\t" << lambda[nc][n] << "\n";
      }
      file.close();
      if (tmin < te[0]){
        std::cout << "Minimum temperature is below that in cooling curve\n"; // " << nc << ": " << coolcurve[nc] << ".\n";
        //std::cout << " tmin = " << tmin << " (K); te[0] = " << te[0] << " (K). Aborting!\n";
        exit(EXIT_FAILURE);
      }
    } // loop over reading ncool cooling curves

    t_min = te[0];       // K
    t_max = te[ntmax-1]; // K
    logtmax = logt[ntmax-1];
    logtmin = logt[0];
    dlogt = (logtmax-logtmin)/float(ntmax-1);

    // Construct alpha(k) (lambda(k) = lambda(n), t(k) = t(n))
      
    for (int n=0; n < ncool; n++){
      for (int k=0; k < ntmax-1; k++){
        alpha[n][k] = std::log10(lambda[n][k+1]/lambda[n][k])/std::log10(te[k+1]/te[k]);
      }

      // Construct the temporal evolution function (TEF)
      // T_ref is an arbitrary reference temperature, and we will set T_ref = T_max
      
      t_ref = t_max;
      y[n][ntmax-1] = 0.0; // zero time to cool from T_ref to T_ref
      lambda_ref[n] = lambda[n][ntmax-1]; // value of lambda at reference temperature
      fac[n] = lambda_ref[n]/t_ref;

      // Construct the TEF using Eq A6 in Townsend (2009)

      for (int k=ntmax-2; k >= 0; k--){
        if (alpha[n][k] != 1.0){
          y[n][k] = y[n][k+1] - (1.0/(1.0-alpha[n][k]))*fac[n]*(te[k]/lambda[n][k])* 
	                                 (1.0 - std::pow((te[k]/te[k+1]),(alpha[n][k]-1.0)));
        }    
        else{
          y[n][k] = y[n][k+1] - fac[n]*(te[k]/lambda[n][k])*std::log(te[k]/te[k+1]);
        }
      }

      // Write out the TEF
      //std::cout << "TEF, coolcurve n = " << n << "\n";
      //for (int m = 0; m < ntmax; m++){
	      //std::cout << logt[m] << "\t" << lambda[n][m] << "\t" << y[n][m] << "\t" << 1.0/y[n][m] << "\n";
      //}
      //exit(EXIT_SUCCESS);
    } // loop over number of cooling curves
         
    if (Globals::my_rank == 0){
      std::cout << "Finished first RadiateExact!\n";
    }
    first = false;
  }

  AthenaArray<Real> dei(pmb->ke+1,pmb->je+1,pmb->ie+1);

  // Now loop over cells, calculating the exact cooling rate
  for (int k = pmb->ks; k <= pmb->ke; ++k){
    for (int j = pmb->js; j <= pmb->je; ++j){
      for (int i = pmb->is; i <= pmb->ie; ++i){

        Real rho = cons(IDN, k, j, i);
        Real u1 = cons(IM1, k, j, i) / rho;
        Real u2 = cons(IM2, k, j, i) / rho;
        Real u3 = cons(IM3, k, j, i) / rho;
        Real v = std::sqrt(u1 * u1 + u2 * u2 + u3 * u3);
        Real ke = 0.5 * rho * v * v;
        Real ie = cons(IEN, k, j, i) - ke;
        Real pre = gmma1 * ie;
        Real temp = pre * avgmass / (rho * boltzman);

        Real tempold = temp;
        Real rhomh = rho / massh;

        if (std::isnan(tempold) || std::isinf(tempold)){
          std::cout << "tempold is a NaN or an Inf. Aborting!\n";
          printf("%.3e %.3e %.3e \n",tempold,rho,pre);
          exit(EXIT_SUCCESS);	  
        }

        int cc = 0;                // use coolcurve1 for ALL cooling

 	      Real avgm = avgmass;
	      Real mu = avgm/massh;

        // Exit (with zero cooling) if temp <= te[0]. We use te[0] instead of tmin, 
        // because if tmin<te[0] it won't trigger if t>tmin but <te[0] and the exact 
        // integration will fail.
        if (temp > 1.1*te[0]){

      	  // Calculate y_temp = y(T). This is the value of y at the desired starting 
	        // temperature. Use Eq. A5 to do this. First find nearest temp and therefore 
	        // the value of the index kk. Also calculate lambda_init
     
  	      Real logtemp = std::log10(temp);
	        int kk = (logtemp-logtmin)/dlogt;
 	
    	    if (kk == ntmax-1){
	          y_temp = 0.0;
	          lambda_init[cc] = lambda[cc][ntmax-1];
          }
          else{
	          if (alpha[cc][kk] != 1.0){
	            y_temp = y[cc][kk] + (1.0/(1.0-alpha[cc][kk]))*fac[cc]*(te[kk]/lambda[cc][kk])*(1.0 - std::pow((te[kk]/temp),(alpha[cc][kk]-1.0)));  //Eq A6
            }
            else{
      	      y_temp = y[cc][kk] + fac[cc]*(te[kk]/lambda[cc][kk])*std::log(te[kk]/temp);
            }
	          lambda_init[cc] = lambda[cc][kk]*std::pow((temp/te[kk]),alpha[cc][kk]);
          }

	        // Calculate the single point cooling time. See p188 of "Computing Notes 2017".
          Real t_cool = 1.0/(gmma1*(rho*mu/(massh*boltzman))*lambda_init[cc]/temp); // This assumes that Edot = (rho/mH)^2 lambda(T)
          //if (rank == 0) cout << "t_cool = " << t_cool << " (s)\n";

	        // Calculate the new temperature given a timestep dt

      	  // Townsend's exact integration scheme (Eq 26)
	        // y_new is the value of the square bracket on the RHS
	        y_new = y_temp + (temp/t_ref)*(lambda_ref[cc]/lambda_init[cc])*(dt/t_cool);

      	  // The new temperature is now given using Eq A7 to obtain Y^-1(y_new) = temp_new (see Eq. 26)
	        // We need to construct the inverse of the TEF (i.e. Y^-1 - see Eq. A7)
	        // First obtain the indx of y_new. This means that y_new lies between y(kk) and y(kk+1)

          int indx = ntmax-2;
          bool tempmin = false;
          while ((y_new > y[cc][indx]) && (indx > 0)){
	          indx--;
	        }
	        if (indx == 0) tempmin = true;
	        if (tempmin == false){
            kk = indx;

	          // Now calculate the inverse of the TEF
	          Real fac2 = lambda[cc][kk]/(te[kk]*fac[cc]);
            Real invy;
            if (alpha[cc][kk] != 1.0){
              Real powr = 1.0/(1.0-alpha[cc][kk]);
              invy = te[kk]*std::pow((1.0-(1.0-alpha[cc][kk])*fac2*(y_new - y[cc][kk])),powr);
            }
            else invy = te[kk]*std::exp(-fac2*(y_new-y[cc][kk])); 
            temp_new_town = invy;
	        }
          else temp_new_town = te[0];
        
  	      Real tempnew = std::max(te[0],temp_new_town); // use te[0] instead of tmin
	        tempnew = std::min(tempnew,tmax);

  	      // Calculate change in energy, dei = delta(temp) * density.
	        // dei is positive if the gas is cooling.
          dei(k,j,i) = (tempold - tempnew)*rho;	

      	} // if T > 1.1*te[0]
	      else{
	        // Set temperature to tmin
          dei(k,j,i) = (tempold - tmin)*rho;	
        }

      } // i loop
    }   // j loop
  }     // k loop

  // Restrict cooling at unresolved interfaces
  if (restrictUnresolvedCooling) RestrictCool(pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke,pmb->pmy_mesh->ndim,gmma1,dei,cons);

  // Adjust pressure due to cooling
  AdjustPressureDueToCooling(pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke,gmma1,dei,cons);

  return;
}


//========================================================================================
//! \fn void RadiateHeatCool(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons);
//  \brief Calculating radiative heating and cooling
//========================================================================================
void RadiateHeatCool(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons){

  const int ncool = 1; // number of cooling curves

  static coolingCurve cc[ncool];
  static bool firstHeatCool = true;
  bool restrictUnresolvedCooling = true;

  Real gmma  = pmb->peos->GetGamma();
  Real gmma1 = gmma - 1.0;

  tmin = Twind;
  tmax = 3.0e9;

  //const Real a = grainRadius;                      // grain radius (cm)
  //const Real dens_g = grainBulkDensity;            // (g/cm^3)
  //const Real massD = (4.0/3.0)*pi*pow(a,3)*dens_g; // grain mass (g)

  Real totEdotGas = 0.0;
  Real totEdotDust = 0.0;

  if (firstHeatCool){
    // Specify the cooling curves and the number of temperature bins.
    // The first cooling curve should extend to the same or lower temperature
    // than the second.
    // The cooling_KI02_4.0_CLOUDY_7.6_MEKAL.txt gas cooling rate is lambda*(dens/mH)^2.
    // The dust_lambdaD_solar_a0.1.txt dust cooling rate is lambda_D*np*ne.
    cc[0].coolCurveFile = "cooling_KI02_4.0_CLOUDY_7.6_MEKAL.txt";
    cc[0].ntmax = 161;
    //#ifdef DUST
    //    cc[1].coolCurveFile = "dust_lambdaD_solar_a0.1.txt";
    //    cc[1].ntmax = 101;
    //#endif

    // Read in each cooling curve. It is assumed that the temperature binning is
    // uniform in log space i.e. dlog(T) is a constant
    for (int nc = 0; nc < ncool; ++nc)
    {
      std::ifstream file(cc[nc].coolCurveFile.c_str());
      if (!file)
      {
        std::cerr << "radiateHeatCool: failed to open file " << cc[nc].coolCurveFile << "\n";
        exit(EXIT_FAILURE);
      }
      for (int n = 0; n < cc[nc].ntmax; ++n)
      {
        Real logt, lambda;
        file >> logt >> lambda;
        cc[nc].logt.push_back(logt);
        cc[nc].lambdac.push_back(lambda);
        cc[nc].te.push_back(pow(10, logt));
        cc[nc].loglambda.push_back(log10(lambda));
        //if (Globals::my_rank == 0) std::cout << "nc = " << nc << "; n = " << n << "\t" << cc[nc].te[n] << "\t" << cc[nc].lambdac[n] << "\n";
      }
      file.close();

      cc[nc].t_min = cc[nc].te[0];                // K
      cc[nc].t_max = cc[nc].te[cc[nc].ntmax - 1]; // K
      cc[nc].logtmax = cc[nc].logt[cc[nc].ntmax - 1];
      cc[nc].logtmin = cc[nc].logt[0];
      cc[nc].dlogt = (cc[nc].logtmax - cc[nc].logtmin) / float(cc[nc].ntmax - 1);
    }

    // Check that tmin is not below the minimum temperature in the first
    // cooling curve
    //if (tmin < cc[0].te[0]){
    //  std::cout << "Minimum temperature is below that in cooling curve\n";
    //  std::cout << " tmin = " << tmin << " (K); te[0] = " << cc[0].te[0] << " (K). Aborting!\n";
    //  exit(EXIT_FAILURE);
    //}

    firstHeatCool = false;
    if (Globals::my_rank == 0){
      std::cout << "Finished firstHeatCool!\n";
    }
  }

  AthenaArray<Real> dei(pmb->ke+1,pmb->je+1,pmb->ie+1);
  
  // Now loop over cells, calculating the heating/cooling rate
  for (int k = pmb->ks; k <= pmb->ke; ++k){
    for (int j = pmb->js; j <= pmb->je; ++j){
      for (int i = pmb->is; i <= pmb->ie; ++i){

        Real rho = cons(IDN, k, j, i);
        Real u1 = cons(IM1, k, j, i) / rho;
        Real u2 = cons(IM2, k, j, i) / rho;
        Real u3 = cons(IM3, k, j, i) / rho;
        Real v = std::sqrt(u1 * u1 + u2 * u2 + u3 * u3);
        Real ke = 0.5 * rho * v * v;
        Real ie = cons(IEN, k, j, i) - ke;
        Real pre = gmma1 * ie;
        Real temp = pre * avgmass / (rho * boltzman);

        Real tempold = temp;
        Real logtemp = std::log10(temp);
        Real rhomh = rho / massh;

#ifdef DUST
        // In this simplest implementation the dust moves with the gas and its
        // mass fraction is given by an advected scalar
        //	z = lg.P0[iqal0][k][j][i];            // dust mass fraction
        //rhod = rho*z;                        // dust mass density (g/cm^3)
        //nD = rhod/massD;                      // grain number density (cm^-3)
        //nH = rho*(10.0/14.0)/massh;          // solar with 10 H per 1 He, avg nucleon mass mu_nu = 14/11.
        ////ntot = 1.1*nH;                        // total nucleon number density
        ////ne = 1.2*nH;                          // electron number density
#endif

        if (std::isnan(tempold) || std::isinf(tempold)){
          std::cout << "tempold is a NaN or an Inf. Aborting!\n";
          printf("%.3e %.3e %.3e \n",tempold,rho,pre);
          exit(EXIT_SUCCESS);	  
        }

        if (temp < 1.5 * Twind){
          // this should prevent wasting time in the unshocked gas
          // set temp to Twind
          temp = Twind;
        }
        else{       
          Real dtint = 0.0;
          Real lcool = 0.0;
          while (dtint < dt){ // "resolve" cooling by restricting it to 10%

            //GammaHeat = 2.0e-26; // erg/s/cm^3

            // Loop over all cooling curves (e.g. gas plus dust)
            Real lambda_cool_nc[ncool] = {0.0};
            for (int nc = 0; nc < ncool; ++nc){
            
              Real lambda_cool = 0.0;
              // If temp <= 1.1*te[0] there is zero cooling. We use te[0] instead of tmin,
              // because if tmin < te[0] it won't trigger if t > tmin but < te[0].
              // Cooling is only calculated if the temperature is
              // within the temperature range of the cooling curve.
              if (temp > 1.1 * cc[nc].te[0] && temp < cc[nc].te[cc[nc].ntmax - 1])
              {
                int kk = int((logtemp - cc[nc].logtmin) / cc[nc].dlogt);
                if (kk == cc[nc].ntmax - 1)
                  lambda_cool = cc[nc].lambdac[cc[nc].ntmax - 1];
                else
                {
                  // temperature lies between logt[kk] and logt[kk+1]
                  Real grad = (cc[nc].loglambda[kk + 1] - cc[nc].loglambda[kk]) / cc[nc].dlogt;
                  lambda_cool = std::pow(10, cc[nc].loglambda[kk] + grad * (logtemp - cc[nc].logt[kk]));
                }
                //if ((temp > 1.0e5)&&(nc == 1)){
                //  cout << "temp = " << temp << "; kk = " << kk << "; lambda_cool = " << lambda_cool << "\n";
                //  cout << "te[0] = " << cc[nc].te[0] << "\n";
                //  cout << "te[ntmax-1] = " << cc[nc].te[cc[nc].ntmax-1] << "\n";
                //  cout << "logtmin = " << cc[nc].logtmin << "\n";
                //  cout << "dlogt = " << cc[nc].dlogt << "\n";
                //  quit();
                //}

                //if (Globals::my_rank == 0 && tempold > 1.0e7){
                //  std::cout << "kk = " << kk << "; ntmax = " << cc[nc].ntmax << "; lambda_cool = " << lambda_cool << "\n";
                //}
              }
              lambda_cool_nc[nc] = lambda_cool;
            }

            // Calculate cooling rate due to gas
            Real edotGas = rhomh * rhomh * lambda_cool_nc[0]; // erg/cm^-3/s
            Real total_cool = edotGas;
            //if (Globals::my_rank == 0 && tempold > 1.0e7){
            //  std::cout << "edotGas = " << edotGas << "; lambda_cool_nc[0] = " << lambda_cool_nc[0] << "\n";
            //}

#ifdef DUST
            // Calculate cooling rate due to dust
            //std::cout << "Hi";
            //Real edotDust = nH * nD * lambda_cool_nc[1]; // due to dust
            //if (temp > 1.0e5){
            //   cout << "nH = " << nH << "; nD = " << nD << "; lambda_D = " << lambda_cool_nc[1] << "\n";
            //   quit();
            //}
            //total_cool += edotDust;
#endif

            Real Eint = pre / gmma1;
            Real t_cool = Eint / total_cool;
            Real dtcool = 0.1 * std::abs(t_cool); // maximum timestep to sample the cooling curve
            if ((dtint + dtcool) > dt)
              dtcool = dt - dtint;
            dtint += dtcool;
            lcool += v*dtcool;

            // Calculate new temperature, and update pressure
            Real dEint = -total_cool * dtcool;
            Real tempnew = temp * (Eint + dEint) / Eint;

	          if (tempnew < tmin){ // Exit out of cooling sub-cycling
	            temp = tmin;
	            break;
	          }
	    
            tempnew = std::min(tempnew, tmax);
            pre *= tempnew / temp;
            temp = tempnew;
            
            // Set a limit to the cooling (DON'T DO THIS unless there is a very good reason as it changes the physics!)
            // Real restrictfrac = 0.97;
            // if (temp <= restrictfrac * tempold)//0.97
            // { // maximum cooling allowed
            //  temp = restrictfrac * tempold;
            //  break;
          //  }

            logtemp = std::log10(temp);

            //if (Globals::my_rank == 0 && tempold > 1.0e7){
            //  std::cout << "tempold = " << tempold << "; temp = " << temp << "\n";
            //  std::cout << "dt = " << dt << "; dtcool = " << dtcool << "; total_cool = " << total_cool << "; Eint = " << Eint << "; dEint = " << dEint << "\n";
            //}

            // Monitor total energy loss rate
            //totEdotGas += edotGas*vol;               // erg/s
#ifdef DUST
            //totEdotDust += edotDust*vol;
#endif
          } // end of cooling sub-cycling
        }   // cool?

	      // Calculate change in energy, dei = delta(temp) * density.
	      // dei is positive if the gas is cooling.
        dei(k,j,i) = (tempold - temp)*rho;	
      }
    }
  }

  //if (ncycle%1000 == 0){
  //  cout << "totEdotGas = " << totEdotGas << "; totEdotDust = " << totEdotDust << " (erg/s)\n";
  //}

  // Restrict cooling at unresolved interfaces
  if (restrictUnresolvedCooling) RestrictCool(pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke,pmb->pmy_mesh->ndim,gmma1,dei,cons);

  // Adjust pressure due to cooling
  AdjustPressureDueToCooling(pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke,gmma1,dei,cons);
}



/*!  \brief Restrict the cooling rate at unresolved interfaces between hot 
 *          diffuse gas and cold dense gas.
 *
 *   Replace deltaE with minimum of neighboring deltaEs at the interface.
 *   Updates dei, which is positive if the gas is cooling.
 *
 *   \author Julian Pittard (Original version 13.09.11)
 *   \version 1.0-stable (Evenstar):
 *   \date Last modified: 13.09.11 (JMP)
 */
void RestrictCool(int is,int ie,int js,int je,int ks,int ke,int nd,Real gmma1,AthenaArray<Real> &dei,const AthenaArray<Real> &cons){

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
void AdjustPressureDueToCooling(int is,int ie,int js,int je,int ks,int ke,Real gmma1,AthenaArray<Real> &dei,AthenaArray<Real> &cons){
  
  for (int k = ks; k <= ke; k++){
    for (int j = js; j <= je; j++){
      for (int i = is; i <= ie; i++){
        
        Real rho = cons(IDN,k,j,i);
	      Real u1  = cons(IM1,k,j,i)/rho; 
	      Real u2  = cons(IM2,k,j,i)/rho; 
	      Real u3  = cons(IM3,k,j,i)/rho;
	      Real ke = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
	      Real pre = (cons(IEN, k, j, i) - ke)*gmma1;

        //col  = prim[iqal0][k][j][i];   // (=1.0 for solar, 0.0 for WC)

	      //if (col > 0.5) avgm = stavgm[0][0];
        //else           avgm = stavgm[1][0]; 
        Real avgm = avgmass;
	
        //if (col <= 0.5) dei[k][j][i] = 0.0;
     
  	    Real const_1 = avgm/boltzman;

	      Real tmpold = const_1 * pre / rho;
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

// void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons){

//   const Real dens_g = grainBulkDensity;               // (g/cm^3)
//   const Real A = 12.0;    // Carbon dust grains
//   const Real eps_a = 0.1; // probability of sticking

//   Real gmma  = pmb->peos->GetGamma();
//   Real gmma1 = gmma - 1.0;
  
//   for (int k=pmb->ks; k<=pmb->ke; ++k) {
//     Real zc = pmb->pcoord->x3v(k) - zpos1;
//     Real zc2 = zc*zc;
//     for (int j=pmb->js; j<=pmb->je; ++j) {
//       Real yc = pmb->pcoord->x2v(j) - ypos1;
//       Real yc2 = yc*yc;
//       for (int i=pmb->is; i<=pmb->ie; ++i) {
//         //cout << "i = " << i << "; j = " << j << "; k = " << k << "\n";
//         Real xc = pmb->pcoord->x1v(i) - xpos1;
// 	      Real xc2 = xc*xc;
// 	      Real r2 = xc2 + yc2 + zc2;

//         bool passedBasicCheck = true;

// 	      Real rho = cons(IDN,k,j,i);
// 	      Real nH = rho*(10.0/14.0)/massh;  // solar with 10 H per 1 He, avg nucleon mass mu_nu = 14/11. 
//         Real ntot = 1.1*nH;                        // total nucleon number density
//         Real z = pmb->pscalars->s(1,k,j,i)/rho;    // dust mass fraction
//         Real rhod = rho*z;                         // dust mass density (g/cm^3)

//         Real a = (pmb->pscalars->s(2,k,j,i)/rho)*1.0e-4; // grain radius (cm)
//         Real massD = (4.0/3.0)*pi*std::pow(a,3)*dens_g; // grain mass (g)

// 	      Real nD = rhod/massD;                      // grain number density (cm^-3)

// 	      Real u1  = cons(IM1,k,j,i)/rho; 
// 	      Real u2  = cons(IM2,k,j,i)/rho; 
// 	      Real u3  = cons(IM3,k,j,i)/rho;
// 	      Real ke = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
// 	      Real pre = (cons(IEN, k, j, i) - ke)*gmma1;

//         Real col = pmb->pscalars->s(0,k,j,i)/rho;  // wind colour (1.0 for primary wind, 0.0 for secondary wind)
//         Real cols[2];
//         cols[0] = col;
//         cols[1] = 1.0 - col;
	
//         //col  = prim[iqal0][k][j][i];   // (=1.0 for solar, 0.0 for WC)
//       	//if (col > 0.5) avgm = stavgm[0][0];
//         //else           avgm = stavgm[1][0]; 
//         Real avgm = avgmass;     
// 	      Real temp = avgm * pre / (rho*boltzman);

//       	// Determine overdensity from smooth WR wind
// 	      Real rho_smooth =  mdot1/(4.0*pi*r2*vinf1);
	
//         Real rhod_dot = 0.0;
//         Real dadt = 0.0;

//         // Perform sanity check series
//         if (col < 0.0 || col > 1.0) passedBasicCheck = false; // Assure winds have sensible colour

//         // Perform a quick check to see if grains are sensible size
//         Real aThreshold = 0.01 * minimumGrainRadiusMicrons;
//         Real zThreshold = 0.01 * minimumDustToGasMassRatio;
//         if (a >= aThreshold && z >= zThreshold && passedBasicCheck) {
//           // If grains are too hot, then sputter
//           if (temp > 1.0e6 && z > 0.0){
//             // Dust thermal sputtering
//             Real tau_d = 3.156e17*a/ntot;            // grain destruction time (s)
//             dadt = -a/tau_d;
//             rhod_dot = -1.33e-17*dens_g*a*a*ntot*nD; // dust destruction rate (g cm^-3 s^-1)
//           }
//           // If grains are warm, accrete
//           if (temp < 1.4e4){
//             // Calculate the average carbon mass fraction in the wind
//             Real carbonMassFrac = 0.0;
//             for (int nw = 0; nw < 2; nw++) {        
//               // Find per-wind carbon mass fraction, this is multiplied by the wind
//               // "colour" in order to determine the mass fraction   
//               Real windCarbonMassFrac = cols[nw] * massFrac[nw][2];
//               // Accululate, such that cmf = sum(windmfrac * colour)
//               carbonMassFrac += windCarbonMassFrac;
//             }
//             // Calculate gas RMS velocity
//             Real wa  = std::sqrt(3.0*boltzman*temp/(A*massh));
//             // Calculate grain radius increase due to impinging carbon atoms
//             Real rhoC = rho * carbonMassFrac;  // Gas density of carbon in wind (g cm^-3)
//             dadt      = 0.25*eps_a*rhoC*wa/dens_g;  // Grain radius increase (cm)
//             // if (dadt > 1.0) printf("%.3e %.3e %.3e %.3e %.3e %.3e %.3e \n",dadt,rho,rhoC,cols[0],cols[1],col,carbonMassFrac);

//             // Calculate associated density incrase
//             rhod_dot = 4.0*PI*SQR(a)*dens_g*nD*dadt;
//           }
//           // If a change in dust has occured, summate and modify scalars & density
//           if (rhod_dot != 0.0){
//             Real drhod = rhod_dot*dt;
//             Real rhodnew = rhod + drhod;
//                  rhodnew = std::max(minimumDustToGasMassRatio*rho,rhodnew); // new dust density
//             Real rhonew = rho + (rhod - rhodnew);                                   // new gas  density (preserving total mass)
//             cons(IDN,k,j,i) = rhonew;
//             // Update the conserved wind colour (as the gas density may have changed)
//             pmb->pscalars->s(0,k,j,i) = col*rhonew;
//             pmb->pscalars->r(1,k,j,i) = rhodnew/rhonew;
//             pmb->pscalars->s(1,k,j,i) = pmb->pscalars->r(1,k,j,i)*rhonew;
//             // Update the grain radius
//             Real anew = std::max(minimumGrainRadiusMicrons,(a + dadt*dt)/1.0e-4); // (microns)
//             pmb->pscalars->r(2,k,j,i) = anew;
//             pmb->pscalars->s(2,k,j,i) = anew*rhonew;
//           }
//         }
//       }
//     }
//   }
//   // Finish up and return
//   return;
// }

// Evolve dust due to sputtering and grain growth
// Multi-wind method developed by JWE
// This is being used instead of original method as WC/OB wind abundances important to dust growth
void EvolveDustMultiWind(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons) {
  // Grab some constants, easier to have them at initialisation rather than in the loop
  const Real minimumGrainRadiusCM = 1e-4*minimumGrainRadiusMicrons;
  // Dust constants
  const Real rho_Gr = grainBulkDensity;
  const Real A      = 12.0; // Atomic mass of carbon dust grains
  const Real eps_a  = 0.1;  // Probability of grain sticking
  const Real m_E[5] = {1.6737736e-24,6.6464737e-24,1.9944735e-23,2.3258673e-23,2.6567629e-23};
  // Thermodynamic constants
  const Real gmma  = pmb->peos->GetGamma(); // Get heat capacity ratio
  const Real gmma1 = gmma - 1.0;            // gamma-1 always useful to have around
  // Initialise arrays
  Real dist2[2][3]; // x y and z distance components from WR and OB stars (cm)
  Real r2[2];       // r^2 distance from WR and OB stars (cm^2)
  Real r[2];        // r distance from WR and OB stars (cm)
  Real n_E[5];
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    // Calculate z distance from stars
    Real zc     = pmb->pcoord->x3v(k); // Z position of cell in simulation domain (cm)
    dist2[0][2] = SQR(zc - zpos1);     // Calculate z distance to WR star
    dist2[1][2] = SQR(zc - zpos2);     // Calculate z distance to OB star 
    for (int j=pmb->js; j<=pmb->je; ++j) {
      // Calculate y distance from star 1
      Real yc = pmb->pcoord->x2v(j);
      dist2[0][1] = SQR(yc - ypos1);
      dist2[1][1] = SQR(yc - ypos2);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real xc = pmb->pcoord->x1v(i); // x component of distance to star 1
	      dist2[0][0] = SQR(yc - ypos1);
        dist2[1][0] = SQR(yc - ypos2);
        // Calculate distances to each star
        for (int s = 0; s < 2; s++) {
          r2[s] = dist2[s][0] + dist2[s][1] + dist2[s][2];
          r[s]  = sqrt(r2[s]);  //TODO check if this is needed
        }
        // Import conserved variables
	      Real rho = cons(IDN,k,j,i);     // Dust density (g/cm^3)
        Real ien = cons(IEN,k,j,i);     // Internal energy (erg) 
        Real u1  = cons(IM1,k,j,i)/rho; // ux gas velocity component (cm/s)
	      Real u2  = cons(IM2,k,j,i)/rho; // uy gas velocity component (cm/s)
	      Real u3  = cons(IM3,k,j,i)/rho; // uz gas velocity component (cm/s)
        // Import scalars to function
        Real col = pmb->pscalars->s(0,k,j,i)/rho; // wind colour (1.0 for primary wind, 0.0 for secondary wind)
        Real z   = pmb->pscalars->s(1,k,j,i)/rho; // dust mass fraction
        Real a   = pmb->pscalars->s(2,k,j,i)/rho; // grain radius (micron)
             a  *= 1.0e-4;                        // Convert grain radius to CGS (cm)
        
        // if (a > 1.1*minimumGrainRadiusCM) printf("%.3e %.3e\n",z,a);
        // Calculate gas properties
        Real v2  = SQR(u1) + SQR(u2) + SQR(u3); // Gas scalar velocity (cm/s)
        Real ke  = 0.5 * rho * v2;              // Kinetic energy of gas (erg) 
        Real pre = (ien - ke) * gmma1;          // Gas pressure (Ba)
        // Calculate wind abundances
        if (col > 1.1 && col < -0.1) printf("COL %.3e\n",col);
        Real cols[]  = {0.0,0.0}; // Array containing wind fractions
             cols[0] = col;       // WR wind fraction
             cols[1] = 1.0 - col; // OB wind fraction

        // if (col > 1.1) std::cout<<col<<"\n";
        // if (col < 0.99 && col > 0.01) std::cout << col << "\n";
        // Calculate carbon density in wind
        Real C_mass_frac = 0.0; // Mass fraction of carbon, anywhere from 0 to 1, summated across winds
        for (int w = 0; w < 2; w++) {
          Real carbon_abundance = massFrac[w][2];
          Real wind_C_mass_frac = cols[w] * massFrac[w][2];
          C_mass_frac += wind_C_mass_frac;
        }
        Real rho_C = rho * C_mass_frac; // Gas density for carbon (g/cm^3)
        // Quick check to see if carbon density is not an unrealistic value
        if (rho_C > rho) rho_C = rho;
        if (rho_C < 0.0) rho_C = 0.0;
        // Calculate number density for wind
        Real nT = 0.0; // Total number density (cm^-3)
        for (int w = 0; w < 2; w++) {
          Real nT_w = 0.0; // Individual wind number density (cm^-3)
          for (int a = 0; a < 5; a++) {
            // Calculate element number density
            n_E[a] = rho * massFrac[w][a] / m_E[a];
            // Add to total number density
            nT_w += n_E[a];
          }
          // Scale number density for wind for wind fraction
          nT += nT_w * cols[w];
          // if (cols[w] < -0.001 or cols[w] > 1.001) {
          //   printf("%.3e %.3e %.3e %.3e\n",nT,nT_w,cols[w],col);
          // }
        }
        //TODO check to see if this calculation is right
        // Calculate the average gas particle mass
        Real avgm = rho / nT; 
        // Calculate gas temperature
        Real T   = (avgm * pre)/(rho * boltzman);  // Gas temperature (K)
        // Calculate dust grain parameters
        Real rho_D   = rho*z;                              // dust mass density (g/cm^3)
        Real mass_Gr = (4.0/3.0)*pi*a*a*a*rho_Gr;  // grain mass (g)
	      Real nD      = rho_D/mass_Gr;                      // grain number density (cm^-3)
      	// Determine overdensity from smooth WR wind
	      Real rho_smooth =  mdot1/(4.0*pi*r2[0]*vinf1);
        // Initialise variables
   	    Real rho_D_dot = 0.0;
	      Real a_dot     = 0.0;
        printf("%.3e\n",T);
        // Determine if dust needs to be evolved, first, perform sanity check
        if (z > minimumDustToGasMassRatio && a > minimumGrainRadiusCM) {
          // Potentially here would be a good place to add a condition to prevent dust growth in remap zones
          // If temperature is too high, thermally sputter dust
          if (T > 1.0e6) {
            Real tau_d     = (3.156e17*a)/nT; // grain destruction time (s)
                 a_dot     = -a/tau_d;
                 rho_D_dot = -1.33e-17*rho_Gr*a*a*nT*nD; 
          }
          // If temperature is between 1e4 and 1.4e4, begin growing grains
          else if (T < 1.4e4) {
            Real wa        = sqrt((3.0*boltzman*T)/(A*massh)); // Thermal velocity of carbon atoms
                 a_dot     = (0.25*eps_a*rho_C*wa)/rho_Gr;     // Growth rate of grains
                 rho_D_dot = 4.0*PI*SQR(a)*rho_Gr*nD*a_dot;    // Extrapolate to find density increase
          }
        }
        // If dust has evolved, update scalars
        if (rho_D_dot != 0.0) {
          Real minimum_z = rho*minimumDustToGasMassRatio;
          Real d_rho_D   = rho_D_dot*dt; // Calculate total grain growth over timestep
          // Calculate new dust density and new gas density, preserving mass
          Real rho_D_new = rho_D+d_rho_D;
               rho_D_new = std::max(minimum_z,rho_D_new); 
          Real rho_new   = rho + (rho_D-rho_D_new); // Subtract change in dust density from density
          Real z_new     = rho_D_new/rho_new;       // Calculate new value for dust-to-gas mass ratio
          // Update conserved density
          cons(IDN,k,j,i) = rho_new;
          // Update the conserved wind colour (as the gas density may have changed)
          pmb->pscalars->s(0,k,j,i) = col*rho_new;
          // Update the dust to gas mass ratio
          pmb->pscalars->r(1,k,j,i) = z_new;
          pmb->pscalars->s(1,k,j,i) = z_new*rho_new;
          // Now update the grain radius
          Real d_a    = a_dot*dt; // Calculate total average radius increase over timestep
          Real a_new  = a+d_a;    // Calculate new grain radius
               a_new *= 1e4;      // Convert back to microns
          Real anew = std::max(minimumGrainRadiusMicrons,a_new); // (microns)
          pmb->pscalars->r(2,k,j,i) = anew;
          pmb->pscalars->s(2,k,j,i) = anew*rho_new;

          if (rho > 2.0*rho_smooth && col > 0.5 && a_dot > 0.0){
	          std::cout << "Growing grains inside WCR...\n";
	          std::cout << "dadt = " << a_dot << "; nD = " << nD << " (cm^-3); rhod_dot = " << rho_D_dot << "\n";
	          std::cout << "Aborting!\n";
	          exit(EXIT_SUCCESS);
	        }
        }
      }
    }
  }
  return;
}


// Evolve the dust (e.g. due to sputtering and grain growth).
// Sputter the dust using the Draine & Salpeter (1979) prescription.
// For T > 1e6 K, the dust lifetime tau_d = 1e6 (a/n) yr (a in microns).
// where a is the grain radius and n the gas nucleon number density.
// See p156-157 of "WBB" book for further details. Sputtered atoms
// contribute to the gas density (and if are co-moving with the gas
// the gas maintains its bulk velocity and pressure).
// Grain growth occurs if T < 1.5e4 K (this gas is assumed to be cool enough
// to form dust). See p20 of CWB Book 2020.
// JMP 22/11/17 - Correctly working.
//
// *** NOTE: In this and DustCreationRateInWCR() I need to replace grainRadius
//           with an advected scalar value...  ****
//
void EvolveDust(MeshBlock *pmb, const Real dt, AthenaArray<Real> &cons){

  const Real dens_g = grainBulkDensity;               // (g/cm^3)
  const Real A = 12.0;    // Carbon dust grains
  const Real eps_a = 0.1; // probability of sticking

  Real gmma  = pmb->peos->GetGamma();
  Real gmma1 = gmma - 1.0;
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real zc = pmb->pcoord->x3v(k) - zpos1;
    Real zc2 = zc*zc;
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real yc = pmb->pcoord->x2v(j) - ypos1;
      Real yc2 = yc*yc;
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //cout << "i = " << i << "; j = " << j << "; k = " << k << "\n";
        Real xc = pmb->pcoord->x1v(i) - xpos1;
	      Real xc2 = xc*xc;
	      Real r2 = xc2 + yc2 + zc2;

	      Real rho = cons(IDN,k,j,i);
	      Real nH = rho*(10.0/14.0)/massh;           // solar with 10 H per 1 He, avg nucleon mass mu_nu = 14/11. 
        Real ntot = 1.1*nH;                        // total nucleon number density
        Real z = pmb->pscalars->s(1,k,j,i)/rho;    // dust mass fraction
        Real rhod = rho*z;                         // dust mass density (g/cm^3)

        Real a = (pmb->pscalars->s(2,k,j,i)/rho)*1.0e-4; // grain radius (cm)
        Real massD = (4.0/3.0)*pi*std::pow(a,3)*dens_g; // grain mass (g)

	      Real nD = rhod/massD;                      // grain number density (cm^-3)

        Real u1  = cons(IM1,k,j,i)/rho; 
	      Real u2  = cons(IM2,k,j,i)/rho; 
	      Real u3  = cons(IM3,k,j,i)/rho;
	      Real ke = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
	      Real pre = (cons(IEN, k, j, i) - ke)*gmma1;

        Real col = pmb->pscalars->s(0,k,j,i)/rho;  // wind colour (1.0 for primary wind, 0.0 for secondary wind)
	
        //col  = prim[iqal0][k][j][i];   // (=1.0 for solar, 0.0 for WC)
	      //if (col > 0.5) avgm = stavgm[0][0];
        //else           avgm = stavgm[1][0]; 
        Real avgm = avgmass;     
	      Real temp = avgm * pre / (rho*boltzman);

      	// Determine overdensity from smooth WR wind
	      Real rho_smooth =  mdot1/(4.0*pi*r2*vinf1);
	
   	    Real rhod_dot = 0.0;
	      Real dadt = 0.0;
	
  	    if (temp > 1.0e6 && z > 0.0){
	        // Dust thermal sputtering
	        Real tau_d = 3.156e17*a/ntot;            // grain destruction time (s)
	        dadt = -a/tau_d;
	        rhod_dot = -1.33e-17*dens_g*a*a*ntot*nD; // dust destruction rate (g cm^-3 s^-1)
	      }
	      else if (temp < 1.4e4){
	        // Dust growth. Requires some grains to exist otherwise rhod_dot = 0.0	  
	        Real wa = std::sqrt(3.0*boltzman*temp/(A*massh));
	        dadt = 0.25*eps_a*rho*wa/dens_g;
	        rhod_dot = 4.0*pi*a*a*dens_g*nD*dadt;    // dust growth rate (g cm^-3 s^-1)
	        //if (rho > 2.0*rho_smooth && col > 0.5){
	          // In cool WCR
	          //std::cout << "Growing grains inside WCR...\n";
	          //std::cout << "dadt = " << dadt << "; nD = " << nD << " (cm^-3); rhod_dot = " << rhod_dot << "\n";
	          //std::cout << "Aborting!\n";
	          //exit(EXIT_SUCCESS);
	        //}
	      }
	      if (rhod_dot != 0.0){
	        Real drhod = rhod_dot*dt;
	        Real rhodnew = std::max(minimumDustToGasMassRatio*rho,rhod + drhod);    // new dust density
	        Real rhonew = rho + (rhod - rhodnew);                                   // new gas  density (preserving total mass)
	        cons(IDN,k,j,i) = rhonew;
	        // Update the conserved wind colour (as the gas density may have changed)
	        pmb->pscalars->s(0,k,j,i) = col*rhonew;
	        // Update the dust to gas mass ratio
	        pmb->pscalars->r(1,k,j,i) = rhodnew/rhonew;
	        pmb->pscalars->s(1,k,j,i) = pmb->pscalars->r(1,k,j,i)*rhonew;
	        // Update the grain radius
	        Real anew = std::max(minimumGrainRadiusMicrons,(a + dadt*dt)/1.0e-4); // (microns)
	        pmb->pscalars->r(2,k,j,i) = anew;
	        pmb->pscalars->s(2,k,j,i) = anew*rhonew;
	      }
      }
    }
  }
  
  return;
}





Real DustCreationRateInWCR(MeshBlock *pmb, int iout){
  const Real dens_g = grainBulkDensity;               // (g/cm^3)
  const Real A = 12.0;    // Carbon dust grains
  const Real eps_a = 0.1; // probability of sticking
  
  Real gmma  = pmb->peos->GetGamma();
  Real gmma1 = gmma - 1.0;

  AthenaArray<Real>& cons = pmb->phydro->u;
  
  Real dmdustdt_WCR = 0.0;
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real zc = pmb->pcoord->x3v(k) - zpos1;
    Real zc2 = zc*zc;
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real yc = pmb->pcoord->x2v(j) - ypos1;
      Real yc2 = yc*yc;
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real xc = pmb->pcoord->x1v(i) - xpos1;
	      Real xc2 = xc*xc;
	      Real r2 = xc2 + yc2 + zc2;

     	  Real rho = cons(IDN,k,j,i);
	      Real nH = rho*(10.0/14.0)/massh;           // solar with 10 H per 1 He, avg nucleon mass mu_nu = 14/11. 
        Real ntot = 1.1*nH;                        // total nucleon number density
        Real z = pmb->pscalars->s(1,k,j,i)/rho;    // dust mass fraction
        Real rhod = rho*z;                         // dust mass density (g/cm^3)

        Real a = (pmb->pscalars->s(2,k,j,i)/rho)*1.0e-4; // grain radius (cm)
        Real massD = (4.0/3.0)*pi*std::pow(a,3)*dens_g; // grain mass (g)

	      Real nD = rhod/massD;                      // grain number density (cm^-3)

     	  Real u1  = cons(IM1,k,j,i)/rho; 
	      Real u2  = cons(IM2,k,j,i)/rho; 
	      Real u3  = cons(IM3,k,j,i)/rho;
	      Real ke = 0.5*rho*(u1*u1 + u2*u2 + u3*u3);
	      Real pre = (cons(IEN, k, j, i) - ke)*gmma1;

        Real col = pmb->pscalars->s(0,k,j,i)/rho;  // wind colour (1.0 for primary wind, 0.0 for secondary wind)
	
	      //if (col > 0.5) avgm = stavgm[0][0];
        //else           avgm = stavgm[1][0]; 
        Real avgm = avgmass;     
	      Real temp = avgm * pre / (rho*boltzman);

	      // Determine overdensity from smooth WR wind
	      Real rho_smooth =  mdot1/(4.0*pi*r2*vinf1);
      	Real rhod_dot = 0.0;

        if (temp < 1.4e4){
	        // Dust growth. Requires some grains to exist otherwise rhod_dot = 0.0	  
	        Real wa = std::sqrt(3.0*boltzman*temp/(A*massh));
	        Real dadt = 0.25*eps_a*rho*wa/dens_g;
	        rhod_dot = 4.0*pi*a*a*dens_g*nD*dadt;    // dust growth rate (g cm^-3 s^-1)
	        Real r = std::sqrt(r2);
	        if (rho > 2.0*rho_smooth && col > 0.5 && r > remapRadius1){
	          // In cool WCR
            Real vol = pmb->pcoord->GetCellVolume(k, j, i);
	          dmdustdt_WCR += rhod_dot*vol;
	        }
	      }
      }
    }
  }
  return dmdustdt_WCR;
}


/*!  \brief Read in the CAK data files.
 * 
 */
void ReadInCAKdataFiles(){

  // Read in the CAK data files.
  std::string filename1 = "o6v.cak";
  std::string filename2 = "o6v.cak";

  // Skip the comments in the file header. Read the file in line-by-line,
  // and then use line-based parsing using stringstreams.
  cd[0].cakFile = filename1;
  std::ifstream infile(filename1);
  if (!infile){
    std::cerr << "Failed to open file " << filename1 << "\n";
    exit(EXIT_FAILURE);
  }
  std::string line;
  getline(infile, line); // 1st line is comments, so ignore these
  getline(infile, line); // 2nd line is comments, so ignore these
  getline(infile, line); // 3rd line is comments, so ignore these
  getline(infile, line); // 4th line is comments, so ignore these
  getline(infile, line); // 5th line is parameter values
  std::istringstream iss(line);
  if (!(iss >> cd[0].rstar >> cd[0].mstar >> cd[0].teff >> cd[0].sigmae >> cd[0].avgmass >> cd[0].alpha >> cd[0].k  )){
    std::cerr << "Failed to correctly read file " << filename1 << "\n";
    exit(EXIT_FAILURE);
  }

  // Read in the rest of the file which contains the actual data values
  while (getline(infile, line)){
    iss.clear();
    iss.str(line);
    Real a,b,c;
    if (!(iss >> a >> b >> c)){
      std::cerr << "Failed to correctly read file " << filename1 << "\n";
      exit(EXIT_FAILURE);
    }
    cd[0].radius.push_back(a*Rsol);
    cd[0].density.push_back(b);
    cd[0].velocity.push_back(c);
  }
  cd[0].ncakmax = cd[0].radius.size();
  infile.close();
  if (Globals::my_rank == 0) std::cout << "Finished reading in CAK datafile1: " << filename1 << "\n";
  
  cd[1].cakFile = filename2;
  infile.open(filename2);
  if (!infile){
    std::cerr << "Failed to open file " << filename2 << "\n";
    exit(EXIT_FAILURE);
  }
  getline(infile, line); // 1st line is comments, so ignore these
  getline(infile, line); // 2nd line is comments, so ignore these
  getline(infile, line); // 3rd line is comments, so ignore these
  getline(infile, line); // 4th line is comments, so ignore these
  getline(infile, line); // 5th line is parameter values
  iss.clear();
  iss.str(line);
  if (!(iss >> cd[1].rstar >> cd[1].mstar >> cd[1].teff >> cd[1].sigmae >> cd[1].avgmass >> cd[1].alpha >> cd[1].k  )){ 
    std::cerr << "Failed to correctly read file " << filename2 << "\n";
    exit(EXIT_FAILURE);
  }
  // Read in the rest of the file which contains the actual data values
  while (getline(infile, line)){
    iss.clear();
    iss.str(line);
    Real a,b,c;
    if (!(iss >> a >> b >> c)){
      std::cerr << "Failed to correctly read file " << filename2 << "\n";
      exit(EXIT_FAILURE);
    }
    cd[1].radius.push_back(a*Rsol);
    cd[1].density.push_back(b);
    cd[1].velocity.push_back(c);
  }
  cd[1].ncakmax = cd[1].radius.size();
  infile.close();
  if (Globals::my_rank == 0) std::cout << "Finished reading in CAK datafile2: " << filename2 << "\n";

  // Adjust units
  for (int n = 0; n < 2; ++n){
    cd[n].mstar *= Msol;  
    cd[n].rstar *= Rsol;  
    cd[n].avgmass *= massh;  
  }

  // Standard conventional practice is to normalize the CAK parameter k by 
  // a fiducial value of vtherm as shown (see Abbott 1982). NOTE: This does
  // not actually correspond to the real thermal sound speed of the gas

  cd[0].vtherm = std::sqrt(2.0*boltzman*cd[0].teff/massh); // definition of vtherm by Abbott 1982
  cd[1].vtherm = std::sqrt(2.0*boltzman*cd[1].teff/massh); // definition of vtherm by Abbott 1982

  // Set file-scope variables
  mstar1 = cd[0].mstar;
  mstar2 = cd[1].mstar;
  rstar1 = cd[0].rstar;
  rstar2 = cd[1].rstar;
  teff1 = cd[0].teff;
  teff2 = cd[1].teff;
  vtherm1 = cd[0].vtherm;
  vtherm2 = cd[1].vtherm;
  avgmass1 = cd[0].avgmass;
  avgmass2 = cd[1].avgmass;
  sigmae1 = cd[0].sigmae;
  sigmae2 = cd[1].sigmae;
  alpha1 = cd[0].alpha;
  alpha2 = cd[1].alpha;
  k1 = cd[0].k;
  k2 = cd[1].k;

  //for (int n = 0; n < cd[0].radius.size(); ++n){
  //  std::cout << "n = " << n << "; radius = " << cd[0].radius[n] << "; density = " << cd[0].density[n] << "; velocity = " << cd[0].velocity[n] << "\n";
  //}

  // Calculate the resolution requirement at the stars to resolve the radiative driving
  //Real desiredResolution = dsep/Real(minimumNumberOfCellsBetweenStars);
  //maxLevelToRefineTo = std::log2(G0dx/desiredResolution);
  //if (G0dx/std::pow(2.0,maxLevelToRefineTo) > desiredResolution) maxLevelToRefineTo++;
  //maxLevelToRefineTo += G0_level;

  return;
}


/*!  \brief Calculate the forces due to the stars (gravity + radiative driving).
 * 
 */
void StellarForces(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, AthenaArray<Real> &cons){

  //Real gmma  = pmb->peos->GetGamma();
  //Real gmma1 = gmma - 1.0;

  Real xpos[2]={xpos1,xpos2};
  Real ypos[2]={ypos1,ypos2};
  Real zpos[2]={zpos1,zpos2};
  Real xvel[2]={xvel1,xvel2};
  Real yvel[2]={yvel1,yvel2};
  Real mdot[2]={mdot1,mdot2};
  Real vinf[2]={vinf1,vinf2};
  Real scalar[2]={1.0,0.0};
  Real mstar[2]={mstar1,mstar2};
  Real rstar[2]={rstar1,rstar2};
  Real teff4[2] ={std::pow(teff1,4),std::pow(teff2,4)}; 
  Real gam_div_sig[2] = {rstar[0]*rstar[0]*stefb*teff4[0]/(G*mstar[0]*cspeed),rstar[1]*rstar[1]*stefb*teff4[1]/(G*mstar[1]*cspeed)};
  Real sigmae[2]={sigmae1,sigmae2};

  Real dx = pmb->pcoord->dx1v(0); // on this MeshBlock
  Real dy = pmb->pcoord->dx2v(0);
  Real dz = pmb->pcoord->dx3v(0);
  Real oneOverdx = 1.0/dx;
  Real oneOverdy = 1.0/dy;
  Real oneOverdz = 1.0/dz;
  Real halfOverdx = 0.5/dx;
  Real halfOverdy = 0.5/dy;
  Real halfOverdz = 0.5/dz;

  // Now loop through the cells to calculate the gravitational force. Add this as a momentum source term
  // and update the conserved variables.
  for (int k = pmb->ks; k <= pmb->ke; ++k){
    for (int j = pmb->js; j <= pmb->je; ++j){
      for (int i = pmb->is; i <= pmb->ie; ++i){
        Real rho = prim(IDN,k,j,i);
        Real u1 = prim(IVX,k,j,i);
        Real u2 = prim(IVY,k,j,i);
        Real u3 = prim(IVZ,k,j,i);
        Real pre = prim(IPR,k,j,i);

        // Decide which CAK values to use based on wind material
        // TODO: depends on colour variable - we need const AthenaArray<Real> &prim_scalar passed into the 
        //       function - see the Athena++ wiki/Problem-Generators page...
        // Real col = prim
        Real alpha = alpha1;
        Real kcak = k1;
        Real vtherm = vtherm1;
        Real sigmae = sigmae1;
        Real avgmass = avgmass1;

        // Calculate velocity gradients
        Real dvxdx = (prim(IVX,k,j,i+1) - prim(IVX,k,j,i-1))*halfOverdx;
        Real dvydx = (prim(IVY,k,j,i+1) - prim(IVY,k,j,i-1))*halfOverdx;
        Real dvzdx = (prim(IVZ,k,j,i+1) - prim(IVZ,k,j,i-1))*halfOverdx;

        Real dvxdy = (prim(IVX,k,j+1,i) - prim(IVX,k,j-1,i))*halfOverdy;
        Real dvydy = (prim(IVY,k,j+1,i) - prim(IVY,k,j-1,i))*halfOverdy;
        Real dvzdy = (prim(IVZ,k,j+1,i) - prim(IVZ,k,j-1,i))*halfOverdy;

        Real dvxdz = (prim(IVX,k+1,j,i) - prim(IVX,k-1,j,i))*halfOverdz;
        Real dvydz = (prim(IVY,k+1,j,i) - prim(IVY,k-1,j,i))*halfOverdz;
        Real dvzdz = (prim(IVZ,k+1,j,i) - prim(IVZ,k-1,j,i))*halfOverdz;

        Real rhov1dot = 0.0;
        Real rhov2dot = 0.0;
        Real rhov3dot = 0.0;

        for (int n = 0; n < 2; ++n){ // Loop over each star
          Real zc = pmb->pcoord->x3v(k) - zpos[n];
          Real zc2 = zc*zc;
          Real yc = pmb->pcoord->x2v(j) - ypos[n];
          Real yc2 = yc*yc;
          Real xc = pmb->pcoord->x1v(i) - xpos[n];
          Real xc2 = xc*xc;
          Real r2 = xc2 + yc2 + zc2;
          Real r = std::sqrt(r2); // distance of cell to star

          int otherStar = (n+1)%2;
          Real radiusOtherStar = rstar[otherStar];
          Real r0mpx = pmb->pcoord->x1v(i) - xpos[otherStar];
          Real r0mpy = pmb->pcoord->x2v(j) - ypos[otherStar];
          Real r0mpz = pmb->pcoord->x3v(k) - zpos[otherStar];
          Real rOtherStar = std::sqrt(r0mpx*r0mpx + r0mpy*r0mpy + r0mpz*r0mpz);

          //bool desiredQuad = false;
          //if (n == 0 && xc < 0 && yc < 0 && zc < 0) desiredQuad = true;

          if (r > rstar[n]){
            Real xy = std::sqrt(xc2 + yc2);

            // The wind angle is t0 (theta_0; see Fig 3. Cranmer & Owocki 1995)
            // t0 goes from -pi/2.0 (-x axis) to 0 (y-axis) to pi/2.0 (+x axis)
            // (NOTE: this is a new definition - 01/04/03)

            Real cost0 = zc/r;
            Real sint0 = xy/r;

            Real cosp0 = xc/xy;
            Real sinp0 = yc/xy;

            Real gamma_star = gam_div_sig[n]*sigmae;

            // Effective gravity (gravitational force minus radiation force on free electrons)
            Real effg = -(1.0-gamma_star)*G*mstar[n]/r2; // (always negative)
            
            // If the cell temperature is above 10^6 K then assume its totally ionized
            // and that therefore the radiative driving force is zero. Likewise, set a
            // limit to the driving if the velocity becomes too high

            Real temp = (pre/rho)*(avgmass/boltzman);
            Real vel = std::sqrt(u1*u1 + u2*u2 + u3*u3);
            //if ((temp < 1.0e6) && (vel < 4.5e8)){ // This allows gas in the cwb1 sim to reach 4.5e8 cm/s
            if ((temp < 1.0e6) && (vel < vinf[n])){
              // Calculate the line-driving force.
              // Calculate the correct normalization of the line driven force
              // and effective gravity (through gamma_star)

              Real gr = rstar[n]*rstar[n]*stefb*teff4[n]*sigmae/(r2*cspeed);
              Real norm = gr*kcak*std::pow(sigmae*rho*vtherm,-alpha);

              // Calculate the half-opening angle of the star
              Real bad = std::asin(rstar[n]/r);

              // Calculate the angle to the offset rings. Mine are best
              Real theta_inner, theta_outer;
              if (num_off == 4){ 
                theta_inner = 0.7071*bad; // Mine
              }
              else if (num_off == 8){
                theta_inner = 0.424*bad; // Mine
                theta_outer = 0.778*bad; // Mine
                // theta_inner = 0.211*bad; // Stan's
                // theta_outer = 0.789*bad; // Stan's       
              }

              // Calculate the offset force angles 
              Real dphi = pi/2.0;
              Real th_1m[13],ph_1m[13];
              for (int m = 0; m < 4; ++m){
                th_1m[m] = theta_inner;
                ph_1m[m] = pi/4.0 + Real(m)*dphi;
              } 
              if (num_off == 8){
                for (int m = 4; m < 8; ++m){
                  th_1m[m] = theta_outer;
                  ph_1m[m] = pi/4.0 + Real(m-4)*dphi; 
                }
              }
              th_1m[num_off] = 0.0;
              ph_1m[num_off] = 0.0;

              // Initialize the velocity gradients and los values
              Real dvbdl[13] = {0.0};
              Real losx[13], losy[13], losz[13];
              Real losa = 1.0; // magnitude of los vector

              // Loop over the integration points on the disk.
              // We must include the central ray in the force calculation to obtain the
              // correct result when we sum the resolved force from individual rays
              bool block = false;
              for (int m = 0; m <= num_off; ++m){
     
                // Calculate the unit vector components of the ray direction.
                // First calculate the unit vector components of the offset ray 
                // in the x_ddash,y_ddash,z_ddash (x", y", z") coordinate system. 
                // The offset ray points outwards from the star
                // (ie x", y", z" are defined here as positive)

                Real x_ddash = std::cos(ph_1m[m])*std::sin(th_1m[m]);
                Real y_ddash = std::sin(ph_1m[m])*std::sin(th_1m[m]);
                Real z_ddash = std::cos(th_1m[m]);

                // Convert offset ray components from (x", y", z") to (x', y', z')
                // (see p37 of CWB 2008 book)

                Real x_dash = x_ddash*cost0 + z_ddash*sint0;
                Real y_dash = y_ddash;
                Real z_dash = z_ddash*cost0 - x_ddash*sint0;
        
                // Convert ray components from (x', y', z') to (x, y, z)

                losx[m] = x_dash*cosp0 - y_dash*sinp0;
                losy[m] = y_dash*cosp0 + x_dash*sinp0;
                losz[m] = z_dash;

                // Check to see if the ray is blocked by the other star.
                // The equation of the line of sight is given by the vector los (as calculated
                // above) and the point xc,yx,zc (the center of the current cell). Blocking
                // is determined by calculating the distance of the centre of the blocking
                // star from the los - if this is less than the radius of the star, and this
                // star is in front of the other star - blocking occurs. In this case 
                // miss out the radiative force calculations for that particular line 
                // of sight, but not for any other lines of sight which may not be 
                // blocked and with the cell still being influenced by gravity).
 
                Real dx = r0mpy*losz[m] - r0mpz*losy[m];
                Real dy = -r0mpx*losz[m] + r0mpz*losx[m];
                Real dz = r0mpx*losy[m] - r0mpy*losx[m];
                Real d = std::sqrt(dx*dx + dy*dy + dz*dz);

                if ((d <= radiusOtherStar) && (r > dsep) && (r > rOtherStar)){
                  block = true;
                }
                else {
                  // If the ray is not blocked, calculate the velocity gradient for each ray
                  dvbdl[m] = losx[m]*losx[m]*dvxdx + losx[m]*losy[m]*dvydx +
                             losx[m]*losz[m]*dvzdx + losy[m]*losx[m]*dvxdy + 
                             losy[m]*losy[m]*dvydy + losy[m]*losz[m]*dvzdy + 
                             losz[m]*losx[m]*dvxdz + losz[m]*losy[m]*dvydz +
                             losz[m]*losz[m]*dvzdz;
                  dvbdl[m] = std::abs(dvbdl[m]); // This is both correct, and required (otherwise dvbdl**alpha = NAN - see also Lobel & Blomme 2008)                    
                }
              } // offset m loop

              if (block){ // at least one of the rays is blocked by the other star
                          // so calculate the force from each ray individually
                Real wt = 1.0/Real(num_off+1); 
                for (int m = 0; m <= num_off; ++m){
                  Real dvbdlToAlpha = std::pow(dvbdl[m],alpha); 
                  rhov1dot += norm*wt*dvbdlToAlpha*losx[m];
                  rhov2dot += norm*wt*dvbdlToAlpha*losy[m];
                  rhov3dot += norm*wt*dvbdlToAlpha*losz[m];
                }
                // Add gravity
                rhov1dot += effg*cosp0*sint0;
                rhov2dot += effg*sinp0*sint0;
                rhov3dot += effg*cost0;
              }
              else {

                // Perform a gaussian quadrature integration over the stellar disk
                // using the appropriate weightings specified by wt(m).
                // Calculate the dynamically correct estimate of the FDCF

                Real dv_int = 0.0;
                Real wt = 1.0/Real(num_off);
                for (int m = 0; m < num_off; ++m){
                  dv_int += wt*std::pow(dvbdl[m],alpha);
                }
                Real fdcf = dv_int*std::pow(dvbdl[num_off],-alpha);

                Real tcak = sigmae*rho*vtherm/dvbdl[num_off];
                Real Mt = gr*fdcf*kcak*std::pow(tcak,-alpha);

                // Add on effective gravity (ie accounting for continuum radiation force)

                rhov1dot += (Mt+effg)*cosp0*sint0;
                rhov2dot += (Mt+effg)*sinp0*sint0;
                rhov3dot += (Mt+effg)*cost0;

                //if (desiredQuad){
                //  std::cout << "xr = " << cosp0*sint0 << "; yr = " << sinp0*sint0 << "; zr = " << cost0 << "\n";
                //  std::cout << "Mt = " << Mt << "; effg = " << effg << "; rhov1dot = " << rhov1dot << "; rhov2dot = " << rhov2dot << "; rhov3dot = " << rhov3dot << "\n";
                //  exit(EXIT_SUCCESS);
                //}
              } // end of block conditional statement

            } // temp < 1e6 so calculate line-driving force
            else { // only calculate the effective gravity (from gravity and continuum scattering)
              rhov1dot += effg*cosp0*sint0; // this is an acceleration (cm/s^2)
              rhov2dot += effg*sinp0*sint0;
              rhov3dot += effg*cost0;
            }

          } // r > rstar?
        } // star loop

        // The forces change the momentum and energy of the gas in each cell 
        // (via momentum and energy source terms - see Eqs 2 and 3 in Pittard 2009)

        // rhov1dot*rho gives force/unit volume
        // d(rho v)/dt = force/unit volume

        // Force/unit volume = (m a)/unit volume = rho a

        // rhov1dot is an acceleration (cm/s^2)
        // dt*rho*rhov1dot = s (g/cm^3) (cm/s^2) = g/cm^2/s
        // Mtm/unit volume (consIM123) = g cm/s/cm^3 = g/cm^2/s 
        
        cons(IM1,k,j,i) += dt*rho*rhov1dot;
        cons(IM2,k,j,i) += dt*rho*rhov2dot; 
        cons(IM3,k,j,i) += dt*rho*rhov3dot; 

        cons(IEN,k,j,i) += dt*rho*(rhov1dot*u1 + rhov2dot*u2 + rhov3dot*u3); 
      } // i loop
    }   // j loop
  }     // k loop

/*
  AthenaArray<Real> dei(pmb->ke+1,pmb->je+1,pmb->ie+1);
 
 */
  return;  
}


