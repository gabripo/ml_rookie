#!/bin/bash

echo "Stopping all the docker instances..."
docker rm $(docker ps -a -q)