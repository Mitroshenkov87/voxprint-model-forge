; VOXPrint AI Model Forge
; Install folder: C:\Program Files\VOXPrint AI Suite\Model Forge\
; Start menu folder: VOXPrint AI Suite
; GUI: venv\Scripts\voxprint-model-forge.exe  CLI: venv\Scripts\voxprint-forge-cli.exe
; Windows 11 24H2 or newer, 64-bit. The setup holds the app wheels and uv.exe only.
; install.ps1 then builds the Python environments. No GPU libraries or model
; files are inside the setup; models are downloaded by the app on first launch.

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
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') + '"',
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
begin
  if CurStep <> ssPostInstall then
    Exit;
  if not Exec(
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\install.ps1"'),
    ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, ResultCode) then
    ResultCode := 1;
  if ResultCode <> 0 then
  begin
    SuppressibleMsgBox('VOXPrint AI Model Forge setup failed. The log is ' + ExpandConstant('{app}\setup-work\install.log') + '.', mbError, MB_OK, IDOK);
    ExitProcess(ResultCode);
  end;
end;
