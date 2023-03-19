# Overview

A quick project for running custom model using
[TFJS React Native][tfjs-react-native] in an Expo project.

## To run it locally:

```
$ yarn
$ yarn start
```

## To run it with docker :

In `docker` folder, copy/paste `.envdev` to `.env` and set your ipv4 in then run the next shell :

```sh
docker-compose up
```

### Useful commands

Stop the docker container.\
In `docker` folder run :

```sh
docker-compose down
```

Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes.

```sh
docker system prune -a
```

# Project inspired by
https://github.com/tensorflow/tfjs-examples/tree/master/react-native/pose-detection

# TODO
- draws a rectangle when it detects something
