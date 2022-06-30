#!/bin/bash
set -ue


function print_notice() {
    echo -e "\e[1;35m$*\e[m" # magenta
}

function gpu_confirm() {
    local result
    result=`lspci | grep -i nvidia`
    echo $result
}


function main() {
    pyenv local $(printf '%s' $(<.python-version))
    poetry install
    poetry update

    result=`gpu_confirm`

    if [[ $result == "" ]]; then
        print_notice "It seems that you have NO gpus in your machine."
        poetry add torch torchvision
        poetry run poe add-depend-torch
    else
        print_notice "It seems that you have gpus in your machine!"
        poetry run poe force-cuda11
        poetry run poe add-depend-torch
    fi
}

main
