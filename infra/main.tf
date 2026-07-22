terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.5"
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "vm_sku" {
  type    = string
  default = "Standard_NC6s_v3"
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key for VM access"
}

variable "spot_instance" {
  type    = bool
  default = true
}

variable "vm_size_gb" {
  type    = number
  default = 128
}

variable "github_repo_url" {
  type        = string
  description = "GitHub repo URL to clone on the VM"
}

variable "github_pat" {
  type        = string
  default     = ""
  description = "Optional GitHub PAT for private repos"
}

resource "azurerm_resource_group" "adagad" {
  location = var.location
  name     = "rg-adagad-gpu"
}

resource "azurerm_virtual_network" "adagad" {
  location               = azurerm_resource_group.adagad.location
  name                   = "vnet-adagad"
  resource_group_name    = azurerm_resource_group.adagad.name
  address_space          = ["10.0.0.0/16"]
}

resource "azurerm_subnet" "adagad" {
  name                 = "subnet-adagad"
  resource_group_name  = azurerm_resource_group.adagad.name
  virtual_network_name = azurerm_virtual_network.adagad.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_network_security_group" "adagad" {
  location                = azurerm_resource_group.adagad.location
  name                    = "nsg-adagad"
  resource_group_name     = azurerm_resource_group.adagad.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_subnet_network_security_group_association" "adagad" {
  subnet_id                 = azurerm_subnet.adagad.id
  network_security_group_id = azurerm_network_security_group.adagad.id
}

data "azurerm_platform_image" "ubuntu" {
  location  = var.location
  publisher = "Canonical"
  offer     = "ubuntu-22_04-lts"
  sku       = "server-gen2"
}

resource "tls_private_key" "adagad" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "azurerm_public_ip" "adagad" {
  location            = azurerm_resource_group.adagad.location
  name                = "pip-adagad"
  resource_group_name = azurerm_resource_group.adagad.name
  allocation_method   = "Dynamic"
  sku                 = "Basic"
}

resource "azurerm_network_interface" "adagad" {
  location            = azurerm_resource_group.adagad.location
  name                = "nic-adagad"
  resource_group_name = azurerm_resource_group.adagad.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.adagad.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.adagad.id
  }
}

resource "azurerm_linux_virtual_machine" "adagad" {
  location            = azurerm_resource_group.adagad.location
  name                = "vm-adagad"
  resource_group_name = azurerm_resource_group.adagad.name
  size                = var.vm_sku
  admin_username      = var.admin_username

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = var.vm_size_gb
  }

  source_image_reference {
    publisher = data.azurerm_platform_image.ubuntu.publisher
    offer     = data.azurerm_platform_image.ubuntu.offer
    sku       = data.azurerm_platform_image.ubuntu.sku
    version   = data.azurerm_platform_image.ubuntu.version
  }

  spot {
    eviction_policy = "Deallocate"
    max_price       = -1
  }

  provisioner "file" {
    source      = "scripts/bootstrap_gpu.sh"
    destination = "/tmp/bootstrap_gpu.sh"

    connection {
      type        = "ssh"
      host        = azurerm_public_ip.adagad.ip_address
      user        = var.admin_username
      private_key = tls_private_key.adagad.private_key_pem
    }
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/bootstrap_gpu.sh",
      "sudo /tmp/bootstrap_gpu.sh ${var.github_repo_url} ${var.github_pat}"
    ]

    connection {
      type        = "ssh"
      host        = azurerm_public_ip.adagad.ip_address
      user        = var.admin_username
      private_key = tls_private_key.adagad.private_key_pem
    }
  }
}
