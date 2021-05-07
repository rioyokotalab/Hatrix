rm -rf $HOME/dev/sandbox
mkdir -p $HOME/dev/sandbox
cd $HOME/dev/sandbox
git clone --depth 1 --branch dev git@github.com:rioyokotalab/Hatrix.git
cd Hatrix

echo `pwd`
if [[ ${SYSTEM_NAME} = "YOKOTA_LAB" ]]; then
    ybatch scripts/SC_instructions/lab.sh
elif [[ ${SYSTEM_NAME} = "ABCI" ]]; then
    qsub -g gcc50609 scripts/SC_instructions/abci.sh
elif [[ ${SYSTEM_NAME} = "TSUBAME" ]]; then
    qsub -g scripts/SC_instructions/tsubame.sh
fi
