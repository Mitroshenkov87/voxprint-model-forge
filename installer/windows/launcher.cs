using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

internal static class Program
{
    private const string Module = "voxprint_forge.app";
    private const string Title = "VOXPrint AI Model Forge";
    private const int AttachParentProcess = -1;
    private const int StdOutputHandle = -11;
    private const int StdErrorHandle = -12;

    private static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        bool console = WantsConsole(args);
        string py = Path.Combine(dir, "venv", "Scripts", console ? "python.exe" : "pythonw.exe");
        if (!File.Exists(py))
        {
            MessageBox(IntPtr.Zero, Title + " is not set up yet:\r\n" + py, Title, 0x10);
            return 1;
        }
        if (console)
            BindParentConsole();
        var start = new ProcessStartInfo(py, "-I -m " + Module + Tail(Environment.CommandLine));
        start.WorkingDirectory = dir;
        start.UseShellExecute = false;
        if (!console)
        {
            using (Process process = Process.Start(start))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.CreateNoWindow = true;
        start.StandardOutputEncoding = Encoding.UTF8;
        start.StandardErrorEncoding = Encoding.UTF8;
        using (Process process = Process.Start(start))
        {
            process.OutputDataReceived += OnOutput;
            process.ErrorDataReceived += OnError;
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
            return process.ExitCode;
        }
    }

    // --version and --help print and exit. python.exe writes them; pythonw would hide them.
    private static bool WantsConsole(string[] args)
    {
        foreach (string arg in args)
        {
            if (arg == "--version" || arg == "-V" || arg == "--help" || arg == "-h")
                return true;
        }
        return false;
    }

    private static void OnOutput(object sender, DataReceivedEventArgs e)
    {
        if (e.Data != null)
            Console.Out.WriteLine(e.Data);
    }

    private static void OnError(object sender, DataReceivedEventArgs e)
    {
        if (e.Data != null)
            Console.Error.WriteLine(e.Data);
    }

    // A winexe has no console. Attach the parent (cmd) so --version is visible, or open one.
    private static void BindParentConsole()
    {
        if (!AttachConsole(AttachParentProcess))
            AllocConsole();
        var output = new FileStream(new SafeFileHandle(GetStdHandle(StdOutputHandle), false), FileAccess.Write);
        var error = new FileStream(new SafeFileHandle(GetStdHandle(StdErrorHandle), false), FileAccess.Write);
        Console.SetOut(new StreamWriter(output) { AutoFlush = true });
        Console.SetError(new StreamWriter(error) { AutoFlush = true });
    }

    // Everything after argv[0], passed through unchanged.
    private static string Tail(string line)
    {
        int i = 0;
        if (line.StartsWith("\""))
        {
            i = line.IndexOf('"', 1);
            i = i < 0 ? line.Length : i + 1;
        }
        else
        {
            while (i < line.Length && line[i] != ' ' && line[i] != '\t') i++;
        }
        return i < line.Length ? line.Substring(i) : "";
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AttachConsole(int processId);

    [DllImport("kernel32.dll")]
    private static extern bool AllocConsole();

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetStdHandle(int handle);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr owner, string text, string caption, uint type);
}
