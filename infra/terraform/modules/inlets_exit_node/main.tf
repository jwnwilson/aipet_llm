terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "inlets_token" {
  description = "Auth token for the inlets-pro tunnel (written to /etc/inlets/token on the droplet)"
  type        = string
  sensitive   = true
}

variable "inlets_license" {
  description = "inlets-pro license key (written to /etc/inlets/license on the droplet)"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "DigitalOcean region for the exit node"
  type        = string
  default     = "lon1"
}

variable "droplet_size" {
  description = "DigitalOcean droplet size — s-1vcpu-1gb is sufficient for pure TCP relay"
  type        = string
  default     = "s-1vcpu-1gb"
}

# ── Resources ─────────────────────────────────────────────────────────────────

resource "digitalocean_droplet" "inlets_exit" {
  name      = "inlets-exit-node"
  region    = var.region
  size      = var.droplet_size
  image     = "ubuntu-22-04-x64"
  user_data = templatefile("${path.module}/cloud-init.yaml.tpl", {
    inlets_token   = var.inlets_token
    inlets_license = var.inlets_license
  })
  tags = ["inlets", "exit-node"]
}

resource "digitalocean_reserved_ip" "inlets_exit" {
  region = var.region
}

resource "digitalocean_reserved_ip_assignment" "inlets_exit" {
  ip_address = digitalocean_reserved_ip.inlets_exit.ip_address
  droplet_id = digitalocean_droplet.inlets_exit.id
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "reserved_ip" {
  description = "Reserved IP for the inlets exit node — set as INLETS_SERVER_IP GitHub Actions secret and update DNS"
  value       = digitalocean_reserved_ip.inlets_exit.ip_address
}
