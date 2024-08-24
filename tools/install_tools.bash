TOOLS="wget sudo"
echo AUDITWHEEL_POLICY=${AUDITWHEEL_POLICY}
export BASE_POLICY=manylinux
if [ "${AUDITWHEEL_POLICY:0:9}" == "musllinux" ]; then
	export BASE_POLICY=musllinux
fi
echo BASE_POLICY=${BASE_POLICY}
# others may not support yum install, eg. i686/musllinux
if [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
    PACKAGE_MANAGER=yum
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_28" ]; then
	PACKAGE_MANAGER=dnf
elif [ "${BASE_POLICY}" == "musllinux" ]; then
	PACKAGE_MANAGER=apk
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi

if [ "${PACKAGE_MANAGER}" == "yum" ]; then
	yum -y install ${TOOLS}
elif [ "${PACKAGE_MANAGER}" == "apk" ]; then
	apk add --no-cache ${TOOLS}
elif [ "${PACKAGE_MANAGER}" == "dnf" ]; then
	dnf -y install --allowerasing ${TOOLS}
else
	echo "Not implemented"
	exit 1
fi