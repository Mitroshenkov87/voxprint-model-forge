; VOXPrint AI Model Forge
; Install folder: C:\Program Files\VOXPrint AI Suite\Model Forge\
; Start menu folder: VOXPrint AI Suite
; GUI: venv\Scripts\voxprint-model-forge.exe  CLI: venv\Scripts\voxprint-forge-cli.exe
; Windows 11 24H2 or newer, 64-bit. The setup holds the app wheels, the pinned
; bitsandbytes wheel, and uv.exe. install.ps1 then builds the Python environments.
; The .ps1 steps run hidden under PowerShell 7 (pwsh.exe) when it is installed. If it is missing, an
; interactive setup offers to install the pinned official MSI; otherwise Windows PowerShell 5.1 runs them.
; No model files are inside the setup; models are downloaded by the app on first launch.

#define MyAppName "VOXPrint AI Model Forge"
#define MySuiteName "VOXPrint AI Suite"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define MyAppPublisher "Aleksandr Mitroshenkov"

[Setup]
AppId={{CB5C2217-6254-48EA-99F6-BC3D52260BAB}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MySuiteName}\Model Forge
DefaultGroupName={#MySuiteName}
DisableDirPage=yes
DisableProgramGroupPage=yes
UsePreviousAppDir=no
UsePreviousGroup=no
OutputBaseFilename=VoxprintModelForgeSetup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.26100
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\venv\Scripts\voxprint-model-forge.exe
AppVerName={#MyAppName} {#AppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
english.PwshCaption=Installing PowerShell 7 (Microsoft)
russian.PwshCaption=Установка PowerShell 7 (Microsoft)
english.PwshDescription=Setup downloads PowerShell 7 from Microsoft's official release and installs it.
russian.PwshDescription=Установщик скачивает PowerShell 7 из официального выпуска Microsoft и устанавливает его.
english.PwshInstalling=Installing PowerShell 7...
russian.PwshInstalling=Устанавливается PowerShell 7...
english.PwshOffer=PowerShell 7 (Microsoft) is not installed. Setup can download it from Microsoft's official release (about 112 MB) and install it now. This is recommended.%n%nInstall PowerShell 7 now? If you choose No, setup continues with Windows PowerShell 5.1.
russian.PwshOffer=PowerShell 7 (Microsoft) не установлен. Установщик может скачать его из официального выпуска Microsoft (около 112 МБ) и установить сейчас. Это рекомендуется.%n%nУстановить PowerShell 7 сейчас? Если выбрать «Нет», установка продолжится с Windows PowerShell 5.1.
english.PwshDeclined=Setup continues with Windows PowerShell 5.1. We recommend installing PowerShell 7 when you can:%nhttps://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows
russian.PwshDeclined=Установка продолжится с Windows PowerShell 5.1. Рекомендуем установить PowerShell 7, когда будет возможность:%nhttps://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows
english.PwshFailed=PowerShell 7 could not be installed, so setup continues with Windows PowerShell 5.1. We recommend installing PowerShell 7 when you can:%nhttps://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows
russian.PwshFailed=Не удалось установить PowerShell 7, поэтому установка продолжится с Windows PowerShell 5.1. Рекомендуем установить PowerShell 7, когда будет возможность:%nhttps://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows
english.InstallFailed=VOXPrint AI Model Forge setup failed at this step: %1%n%nThe log is %2.
russian.InstallFailed=Не удалось установить VOXPrint AI Model Forge. Шаг: %1%n%nЖурнал: %2.

[Files]
Source: "staging\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Extracted to {tmp} by PrepareToInstall only; not installed.
Source: "migrate-home.ps1"; Flags: dontcopy

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\voxprint-model-forge.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\venv-mivolo"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\setup-work"
Type: dirifempty; Name: "{app}"
; Shared with the other suite apps: removed only when empty.
Type: dirifempty; Name: "{autopf}\{#MySuiteName}"

[Code]
procedure ExitProcess(uExitCode: Cardinal);
  external 'ExitProcess@kernel32.dll stdcall';

// PowerShell 7 LTS x64 MSI from the PowerShell/PowerShell GitHub release; sha256 from its hashes.sha256.
// The installer workflow downloads this pin and checks the hash and the Microsoft Authenticode signature.
const
  PwshMsiUrl = 'https://github.com/PowerShell/PowerShell/releases/download/v7.6.7/PowerShell-7.6.7-win-x64.msi';
  PwshMsiName = 'PowerShell-7.6.7-win-x64.msi';
  PwshMsiSha256 = 'ec575cef9ec56fcd905de0d5eeac9cb7f81a0a65abb87bb5cd0e87fdfe2b8cc0';

var
  DownloadPage: TDownloadWizardPage;
  PwshExe: String;

function FindPwsh: String;
begin
  Result := ExpandConstant('{commonpf64}\PowerShell\7\pwsh.exe');
  if not FileExists(Result) then
    Result := FileSearch('pwsh.exe', GetEnv('PATH'));
end;

// pwsh.exe when present, else Windows PowerShell 5.1.
function ScriptHost: String;
begin
  if PwshExe = '' then
    PwshExe := FindPwsh;
  if PwshExe <> '' then
    Result := PwshExe
  else
    Result := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
  Log('script host: ' + Result);
end;

function ScriptArgs(const Script: String): String;
begin
  Result := '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + Script + '"';
end;

// Download (size and sha256 checked by Inno) and install silently. True when pwsh.exe is then found.
function InstallPwsh: Boolean;
var
  ResultCode: Integer;
begin
  Result := False;
  DownloadPage.Clear;
  DownloadPage.Add(PwshMsiUrl, PwshMsiName, PwshMsiSha256);
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      DownloadPage.SetText(CustomMessage('PwshInstalling'), '');
      if Exec(ExpandConstant('{sys}\msiexec.exe'),
        '/i "' + ExpandConstant('{tmp}\' + PwshMsiName) + '" /qn /norestart ADD_PATH=1',
        '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        Log(Format('PowerShell 7 msiexec exit code %d', [ResultCode]))
      else
        Log('PowerShell 7 msiexec did not start');
      PwshExe := FindPwsh;
      Result := PwshExe <> '';
    except
      Log('PowerShell 7 download or install failed: ' + GetExceptionMessage);
    end;
  finally
    DownloadPage.Hide;
  end;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(CustomMessage('PwshCaption'), CustomMessage('PwshDescription'), nil);
end;

// Interactive only: offer PowerShell 7 when it is missing. A silent setup never prompts.
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID <> wpReady) or WizardSilent then
    Exit;
  PwshExe := FindPwsh;
  if PwshExe <> '' then
    Exit;
  if MsgBox(CustomMessage('PwshOffer'), mbConfirmation, MB_YESNO) = IDYES then
  begin
    if not InstallPwsh then
      MsgBox(CustomMessage('PwshFailed'), mbInformation, MB_OK);
  end
  else
    MsgBox(CustomMessage('PwshDeclined'), mbInformation, MB_OK);
end;

// One-time move of %LOCALAPPDATA%\Voxprint to %LOCALAPPDATA%\VOXPrint AI Suite (voxprint-common 0.2.5).
// It runs before any file is copied and before install.ps1 starts Python, as the signed-in user
// (whose LOCALAPPDATA it is). See migrate-home.ps1 for the rules; exit 10 (both folders exist) is
// a warning, any other non-zero code stops the setup with nothing changed.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  Script: String;
begin
  Result := '';
  ExtractTemporaryFile('migrate-home.ps1');
  Script := ExpandConstant('{tmp}\migrate-home.ps1');
  if not ExecAsOriginalUser(
    ScriptHost,
    ScriptArgs(Script) + ' -AppDir "' + ExpandConstant('{app}') + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    ResultCode := 3;
  Log(Format('home migration exit code %d', [ResultCode]));
  case ResultCode of
    0: ;
    10: SuppressibleMsgBox('Both %LOCALAPPDATA%\Voxprint and %LOCALAPPDATA%\VOXPrint AI Suite exist. ' +
          'Nothing was moved or merged; the apps use VOXPrint AI Suite. Review the old folder and move what you still need.',
          mbInformation, MB_OK, IDOK);
    2: Result := 'Close all VOXPrint programs, then run the setup again. Your data was not moved.';
    4: Result := '%LOCALAPPDATA%\VOXPrint AI Suite is not a plain folder. Nothing was moved.';
  else
    Result := 'Moving %LOCALAPPDATA%\Voxprint to %LOCALAPPDATA%\VOXPrint AI Suite failed; the old folder is unchanged. ' +
      'See %LOCALAPPDATA%\VOXPrint AI Suite\logs\home-migration.log or %TEMP%\voxprint-home-migration.log.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Step: AnsiString;
begin
  if CurStep <> ssPostInstall then
    Exit;
  if not Exec(
    ScriptHost,
    ScriptArgs(ExpandConstant('{app}\install.ps1')),
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    ResultCode := 1;
  if ResultCode <> 0 then
  begin
    // install.ps1 writes the failed step (for example "step 'import torch' timed out after 120 s") here.
    if not LoadStringFromFile(ExpandConstant('{app}\setup-work\install-error.txt'), Step) then
      Step := 'install.ps1';
    SuppressibleMsgBox(FmtMessage(CustomMessage('InstallFailed'), [Trim(String(Step)),
      ExpandConstant('{app}\setup-work\install.log')]), mbError, MB_OK, IDOK);
    ExitProcess(ResultCode);
  end;
end;
