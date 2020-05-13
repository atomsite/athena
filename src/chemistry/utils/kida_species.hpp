#ifndef KIDA_SPECIES_H_
#define KIDA_SPECIES_H_

//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file thermo.hpp
//  \brief definitions for heating and cooling processes
//======================================================================================

// Athena++ classes headers
#include "../../athena.hpp"

//c++ header
#include <sstream>    // stringstream
#include <string>     // string

class KidaSpecies{
  friend class ChemNetwork;
  friend class KidaNetwork;
  public:
    KidaSpecies(std::string line, int index);
    //starts form zero, matches the index in species name in ChemNetwork
    const int index;
    std::string name;
  private:
    static const int natom_ = 13;
    int charge_;
    int atom_count_[natom_];
};

#endif //KIDA_SPECIES_H_