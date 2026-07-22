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
