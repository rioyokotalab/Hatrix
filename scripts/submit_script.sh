mkdir -p $HOME/dev/sandbox
cd $HOME/dev/sandbox
git clone --depth 1 --branch dev git@github.com:rioyokotalab/Hatrix.git
cd Hatrix

if [[ $(uname -n) = "login" ]]; then
    yrun scripts/SC_instructions/lab.sh
elif [[ $(uname -n) = "es1.abci.local" ]]; then
    qsub -g gcc50609 scripts/SC_instructions/abci.sh
fi
