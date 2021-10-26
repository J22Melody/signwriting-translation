#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
configs=$base/configs_opennmt

onmt_build_vocab -config $configs/$1.yaml -n_sample -1 -overwrite