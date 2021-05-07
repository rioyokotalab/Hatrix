rm -rf $HOME/dev/sandbox
mkdir -p $HOME/dev/sandbox
cd $HOME/dev/sandbox
git clone --depth 1 --branch feature/SC_automation git@github.com:rioyokotalab/Hatrix.git
cd Hatrix

if [[ $(uname -n) = "login" ]]; then
    yrun scripts/SC_instructions/lab.sh
elif [[ $(uname -n) == *abci.local ]]; then
    qsub -g gcc50609 scripts/SC_instructions/abci.sh
fi
