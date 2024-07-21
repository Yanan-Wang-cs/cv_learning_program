Generate a Keypair Locally: ssh-keygen -t rsa -q -f "$HOME/.ssh/hyperpod_id_rsa" -N ""
Setup Public Key Authentication: Run cat ~/.ssh/hyperpod_id_rsa.pub and copy the public key to ~/.ssh/authorized_keys on the login node.
Configure SSH Client: Add the following configuration to ~/.ssh/config on your local environment:
Host MBZ-Hyperpod-Cluster
      User <your_username> # Should be User.Name@mbzuai.ac.ae (capital U and capital N)
      ProxyCommand sh -c "aws ssm start-session --target sagemaker-cluster:r2cmo9s2mq3n_login-node-i-0c4d06b3c10c33bcf --region us-west-2 --document-name AWS-StartSSHSession --parameters 'portNumber=%p' IdentityFile ~/.ssh/hyperpod_id_rsa.pub"
NOTE: Make sure to replace the <your_username> with your actual username (should be the same as your email address).

Run this command:
echo export AWS_PROFILE=default >> .bashrc # .zshrc instead of .bashrc if you're on mac
Install Remote SSH Plugin on Visual Studio Code.
Connect to the Host via VS Code: Press Command-Shift-P (Mac) or Control-Shift-P (Windows/Linux) to bring up the menu and select Remote-SSH: Connect to Host. Enter the name of cluster: MBZ-Hyperpod-cluster.
NOTE: Newly opened VS Code sessions do not export the environment variables automatically. Make sure to run aws configure sso when you open a new VS Code session.
