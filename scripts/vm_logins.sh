# Rented-VM logins. Sourced by bootstrap and the canvas scripts.
# FORGE_HOST_KIND=local (the default) does not write token files and does not
# call hf auth login. FORGE_HOST_KIND=rented-vm logs in once and leaves the
# logins for the rental. This file does not enable command tracing and does
# not unset tokens.

persist_rented_vm_logins() {
  set +x
  if [[ "${FORGE_HOST_KIND:-local}" != "rented-vm" ]]; then
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" || -z "${KAGGLE_API_TOKEN:-}" ]]; then
    return 0
  fi
  local hf_bin="${VENV:-}/bin/hf"
  if [[ ! -x "$hf_bin" ]]; then
    printf 'hf is missing at %s. Install the venv before a rented-vm login.\n' "$hf_bin" >&2
    exit 9
  fi
  "$hf_bin" auth login --token "$HF_TOKEN"
  local kaggle_dir="${HOME}/.kaggle"
  mkdir -p "$kaggle_dir"
  chmod 700 "$kaggle_dir"
  local old_umask
  old_umask="$(umask)"
  umask 077
  printf '%s\n' "$KAGGLE_API_TOKEN" > "$kaggle_dir/access_token"
  umask "$old_umask"
  chmod 600 "$kaggle_dir/access_token"
}
