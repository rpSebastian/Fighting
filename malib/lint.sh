#!/bin/bash -ev

{
  black --version | grep -E "20.8b1" > /dev/null
} || {
  echo "Linter requires 'black==20.8b1' !"
  exit 1
}

echo "Running isort..."
isort --skip=__init__.py .

echo "Running black..."
black .

echo "Running flake8..."
if [ -x "$(command -v flake8)" ]; then
  flake8 --count --select=E9,F63,F7,F82 --show-source --statistics .
else
  python -m flake8--count --select=E9,F63,F7,F82 --show-source --statistics .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}