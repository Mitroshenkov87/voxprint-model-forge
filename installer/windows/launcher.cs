using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

internal static class Program
{
    private const string Module = "voxprint_forge.app";
    private const string Title = "VOXPrint AI Model Forge";

    private static int Main()
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string py = Path.Combine(dir, "venv", "Scripts", "pythonw.exe");
        if (!File.Exists(py))
        {
            MessageBox(IntPtr.Zero, Title + " is not set up yet:\r\n" + py, Title, 0x10);
            return 1;
        }
        var start = new ProcessStartInfo(py, "-I -m " + Module + Tail(Environment.CommandLine));
        start.WorkingDirectory = dir;
        start.UseShellExecute = false;
        using (Process process = Process.Start(start))
        {
            process.WaitForExit();
            return process.ExitCode;
        }
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

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr owner, string text, string caption, uint type);
}
