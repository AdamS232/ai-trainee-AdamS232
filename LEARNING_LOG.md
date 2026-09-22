1. The difference between pip and mamba is that pip installs Python packages only, while mamba manages    whole environments. People tend to use mamba for environment/system level stuff, while pip for         Python packages ontop.
   
2. Pinning makes sure all installs are reproducible, so it makes sure that everyone gets the same         versions. Its important because if the library updates later it doesn't silently break or change.

3. It would return False because without "--gpus all", the container can't see the host's GPU at all.
