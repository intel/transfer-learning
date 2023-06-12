# Distributed Training

Here are instructions for using distributed/multinode Training with Intel® Transfer Learning Tool.

## Prerequisites

- Participating nodes should have Intel® oneAPI Base Toolkit installed. Verify the files under `/opt/intel/oneapi`
- Participating nodes should have passwordless SSH setup. Instructions to set up are given below.

### Passwordless SSH setup

- Use an existing (or create an) SSH key pair.

    - Check under `~/.ssh` and see if they exist. If present, make sure they have default names `(id_rsa.pub id_rsa)` and they don't have any passphrase.

    - To remove passphrase, type `ssh-keygen -p [-P old_passphrase] [-N new_passphrase] [-f keyfile]` by replacing `new_passphrase` with a blank space.

- How to create SSH key pair:

    - Get to your .ssh directory `cd ~/.ssh` (if this gives you an error, change the permissions: `chmod u+x ~/.ssh`)

    - Run the command: `ssh-keygen -t rsa`

    - The first prompt will ask you what you want to call your key files `(id_rsa.pub id_rsa)`. Press `<enter>` to use the default key names.

    - The second prompt will ask for passphrase. Do not enter any passphrase, just press `<enter>`.

- Locate the two ssh key pair files in your `.ssh` directory (`id_rsa.pub`, `id_rsa`):

    - Open the Public Key in an editor like vi/vim/nano/pico (this is the `.pub` file)

    - The ending of the public key may say `<your_idsid>@<hostname.domain>`, edit this file to omit the `"@<hostname.domain>"` at the end. The result will be your `idsid` only.

    - Create a file in your .ssh directory called `authorized_keys`

    - Paste your entire public key into this file

    - Make sure your new ssh key pair files AND `authorized_keys` files are read-write only for yourself with no permissions for anyone else `(chmod 600 file1 file2 file3)`

- Test the SSH `ssh <ip_or_hostname.domain>`

IMPORTANT NOTE: You have to make sure the `authorized_keys` file exists on all of the target systems that will participate in running the workload (in your local home dir in your `.ssh` directory) with contents of public key inside as well.
