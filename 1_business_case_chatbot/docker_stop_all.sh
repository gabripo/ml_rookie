#!/bin/bash

echo "Stopping all the docker instances..."
docker stop $(docker ps -a -q)