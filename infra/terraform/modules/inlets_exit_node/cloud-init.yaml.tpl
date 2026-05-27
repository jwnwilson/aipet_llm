#cloud-config
write_files:
  - path: /etc/inlets/token
    permissions: '0600'
    owner: root:root
    content: |
      ${inlets_token}
  - path: /etc/inlets/license
    permissions: '0600'
    owner: root:root
    content: |
      ${inlets_license}
  - path: /etc/systemd/system/inlets-pro.service
    owner: root:root
    content: |
      [Unit]
      Description=inlets-pro TCP server
      After=network.target

      [Service]
      Type=simple
      ExecStart=/usr/local/bin/inlets-pro tcp server --auto-tls --port=8123 --token-file=/etc/inlets/token
      Restart=always
      RestartSec=5

      [Install]
      WantedBy=multi-user.target
runcmd:
  - mkdir -p /etc/inlets
  - curl -sLS https://get.inlets.dev | sh
  - systemctl daemon-reload
  - systemctl enable inlets-pro
  - systemctl start inlets-pro
