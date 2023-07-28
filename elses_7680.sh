#!/bin/bash

mol_folder=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1

# Generate the points for the ELSES matrix.
nx=1
ny=1
nz=1
source_file=$mol_folder/C60_fcc2x2x2_20220727.xyz

fcc_xml_file=$mol_folder/C60_fcc2x2x2_20220727.xml
xml_config_file=$mol_folder/config.xml
