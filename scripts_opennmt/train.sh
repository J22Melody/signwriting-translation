#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
configs=$base/configs_opennmt

onmt_train -config $configs/$1.yaml