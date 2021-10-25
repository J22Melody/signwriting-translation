#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
configs=$base/configs_opennmt

onmt_build_vocab -config $configs/sign2en.yaml -n_sample -1