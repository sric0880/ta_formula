#!/bin/bash
set -e

cd "${GITHUB_WORKSPACE}"/"${PACKAGE_PATH}"
DEPS_DIR=.

TA_LIB_TGZ="ta-lib-0.4.0-src.tar.gz"
TA_LIB_URL="http://prdownloads.sourceforge.net/ta-lib/$TA_LIB_TGZ"

if [[ -d $DEPS_DIR/lib ]]; then
  echo "Already built"
  exit 0
fi
mkdir -p $DEPS_DIR/tmp
wget -O "$DEPS_DIR/tmp/$TA_LIB_TGZ" $TA_LIB_URL
pushd $DEPS_DIR/tmp
tar -zxvf $TA_LIB_TGZ
popd
pushd $DEPS_DIR/tmp/ta-lib
./configure
make install
popd
