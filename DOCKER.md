# Docker Installation

This guide will cover the installation of Docker for Windows using the [WSL](https://learn.microsoft.com/en-us/windows/wsl/) backend.

## Check that your computer meets the necessary hardware requirements

- 64-bit processor with Second Level Address Translation (SLAT)
- 4GB system RAM
- BIOS-level hardware virtualization support must be enabled in the BIOS settings.
- Windows user account with administrator privileges to install software.

## Check that your Windows installation meets the requirements

- Windows 11 64-bit: Home or Pro version 21H2 or higher, or Enterprise or Education version 21H2 or higher.
- Windows 10 64-bit: Home or Pro 21H1 (build 19043) or higher, or Enterprise or Education 20H2 (build 19042) or higher.

To check your Windows version and build number, select **Windows logo key + R**, type **winver**, select **OK**. You can update to the latest Windows version by selecting **Start > Settings > Windows Update > [Check for updates](ms-settings:windowsupdate)**.

## Install WSL

- Open a Powershell and run the following command: `wsl --install`.
- Wait until the installation is finished and restart your computer if necessary.
- Download the Linux kernel update package from the [WSL Docs](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) or [here](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi) (direct link).
- Install the Linux kernel update package and restart your computer if necessary.

More information about installing WSL can be found here: [WSL Docs](https://learn.microsoft.com/en-us/windows/wsl/install).

## Install Docker

- Go to [Docker Docs](https://docs.docker.com/desktop/install/windows-install/) and download the Docker for Windows installer or download it directly from [here](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe) (direct link).
- Install Docker and choose the WSL backend (should be the default selection).
- Restart your computer if necessary.

## Optional: Add non-adminstrator users to docker-users group

- If you want to allow non-administrator users to run Docker you have to add them to the *docker-users* group by running the following command in an administrator Powershell: `net localgroup docker-users "your-user-id" /ADD` or if your user is part of a corporate domain `net localgroup docker-users "DOMAIN\your-user-id" /ADD`.

## Run PIA or PIAWeb

- Start **Docker Desktop**.
- **PIA:** Open a Powershell and run `docker run -v C:\docker_share:/exchange -p 8888:8888 -it michabirklbauer/pia:latest`.
- **PIAWeb:** Open a Powershell and run `docker run -p 8501:8501 michabirklbauer/piaweb:latest`.
- To quit Docker, right-click on the symbol to the right of the taskbar and select **Quit Docker Desktop**.

## Done!
