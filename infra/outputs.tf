output "vm_public_ip" {
  value = azurerm_public_ip.adagad.ip_address
}

output "ssh_command" {
  value = "ssh -i <path_to_private_key> ${var.admin_username}@${azurerm_public_ip.adagad.ip_address}"
}

output "private_key_pem" {
  value     = tls_private_key.adagad.private_key_pem
  sensitive = true
}

output "resource_group_name" {
  value = azurerm_resource_group.adagad.name
}
