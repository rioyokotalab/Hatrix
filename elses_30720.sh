#!/bin/bash

# Generate the points for the ELSES matrix.

mol_folder=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_2x2x1

# Generate the points for the ELSES matrix.
nx=2
ny=2
nz=1
source_file=$mol_folder/C60_fcc2x2x2_disorder_expand_2x2x1_20220912.xyz
fcc_xml_file=$mol_folder/C60_fcc2x2x2_disorder_expand_2x2x1_20220912.xml
xml_config_file=$mol_folder/config.xml
