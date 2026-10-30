# Security

## Reporting a problem

Please **don't open a public issue** for a security problem. Report it privately through GitHub: [Security → Report a vulnerability](https://github.com/darshanvss/cubecheck/security/advisories/new). You'll get a reply within a week.

## What CubeCheck does with your data

The website runs the models on your own device. Camera frames, photos and scrambles are processed in the browser and never sent to any server. The site loads its models and ONNX Runtime from its own domain, not from third-party servers.

Something that would break this, such as a way for a frame or a scramble to leave the device, counts as a security problem.
