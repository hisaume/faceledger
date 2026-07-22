# Fix RetinaFace detection and alignment as part of the vector profile

Faceledger 1.x uses RetinaFace detection with alignment enabled and does not
offer detector or alignment overrides. DeepFace's published Facenet512 cosine
benchmark favors this slower profile over its OpenCV default, and fixing it
keeps persisted vectors compatible without encoding additional extraction
settings in every cache filename; changing the profile requires an intentional
cache-compatibility review and rebuild guidance.
