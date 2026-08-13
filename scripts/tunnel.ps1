<#
    scripts/tunnel.ps1 — túnel SSH diário do launcher (cliente Windows).

    Uso:
      .\scripts\tunnel.ps1 -Hostname <ip-do-host> -User <usuario>
      $env:LLM_LAUNCHER_HOST = "<ip>"; .\scripts\tunnel.ps1
      $env:LLM_LAUNCHER_SSH_USER = "<user>"; .\scripts/tunnel.ps1

    Abre o encaminhamento canônico dos dois serviços para loopback local:
    backend na 8420 e llama-server na 8421. Nenhum IP, hostname ou usuário
    pessoal está embutido — este script vai para o repositório público.

    Requisito: cliente OpenSSH do Windows instalado e chave com acesso SSH
    ao host do backend. Encerre o túnel com Ctrl+C.
#>
param(
    [string]$Hostname = $env:LLM_LAUNCHER_HOST,
    [string]$User = $env:LLM_LAUNCHER_SSH_USER
)

if ([string]::IsNullOrWhiteSpace($Hostname)) {
    Write-Error "Host nao informado. Passe -Hostname <ip> ou defina LLM_LAUNCHER_HOST."
    exit 1
}
if ([string]::IsNullOrWhiteSpace($User)) {
    Write-Error "Usuario SSH nao informado. Passe -User <usuario> ou defina LLM_LAUNCHER_SSH_USER."
    exit 1
}

Write-Host "Abrindo tunel para ${User}@${Hostname} (8420 e 8421 em loopback)..."

ssh -N -L 8420:127.0.0.1:8420 -L 8421:127.0.0.1:8421 "${User}@${Hostname}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Tunel ativo. Abra http://127.0.0.1:8420"
}
exit $LASTEXITCODE
