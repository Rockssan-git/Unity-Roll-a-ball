using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

[Serializable]
public class PythonInput
{
    public float[] values;
}

[Serializable]
public class PythonOutput
{
    public int count;
    public float sum;
    public float mean;
    public string message;
}

public class PythonRunner : MonoBehaviour
{
    [Tooltip("Python executable of the venv that has your packages installed.")]
    [SerializeField] private string pythonExecutable = "/home/julien-rsbrg/venv/venv_RL_unity/bin/python";

    [Tooltip("Script path, relative to Assets/StreamingAssets.")]
    [SerializeField] private string scriptRelativePath = "Python/dummy_script.py";

    [SerializeField] private float[] testValues = { 1f, 2.5f, 4f };

    // Test from the component's ⋮ menu in the Inspector during Play mode.
    // (No longer run automatically at Start.) from the component's ⋮ menu in the Inspector during Play mode.
    [ContextMenu("Run Python test")]
    public void RunTest()
    {
        var input = new PythonInput { values = testValues };
        PythonOutput output = RunPython<PythonInput, PythonOutput>(input);
        if (output != null)
            Debug.Log($"Python says: {output.message} | count={output.count}, sum={output.sum}, mean={output.mean}");
    }

    /// <summary>
    /// Runs the script once: sends `input` as JSON on stdin, parses stdout as JSON.
    /// Blocks until the script finishes. Returns default(TOut) on failure.
    /// scriptOverride: another script (relative to StreamingAssets) instead of scriptRelativePath.
    /// </summary>
    public TOut RunPython<TIn, TOut>(TIn input, string scriptOverride = null)
    {
        string relativePath = string.IsNullOrEmpty(scriptOverride) ? scriptRelativePath : scriptOverride;
        string scriptPath = Path.Combine(Application.streamingAssetsPath, relativePath);
        if (!File.Exists(scriptPath))
        {
            Debug.LogError($"Python script not found: {scriptPath}");
            return default;
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonExecutable,
            Arguments = $"\"{scriptPath}\"",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            // UTF-8 both ways so accented text (French questions/answers) survives;
            // no byte-order mark, which would break JSON parsing.
            StandardInputEncoding = new UTF8Encoding(false),
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false),
        };

        try
        {
            using var process = Process.Start(psi);

            // Send the input, then close stdin so the script's sys.stdin.read() returns.
            process.StandardInput.Write(JsonUtility.ToJson(input));
            process.StandardInput.Close();

            // Read stderr asynchronously so a full stderr buffer can't block stdout.
            var stderrTask = process.StandardError.ReadToEndAsync();
            string stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            string stderr = stderrTask.Result;

            if (process.ExitCode != 0)
            {
                Debug.LogError($"Python exited with code {process.ExitCode}:\n{stderr}");
                return default;
            }
            if (!string.IsNullOrWhiteSpace(stderr))
                Debug.LogWarning($"Python stderr:\n{stderr}");

            return JsonUtility.FromJson<TOut>(stdout);
        }
        catch (Exception e)
        {
            Debug.LogError($"Could not run Python ({pythonExecutable}): {e.Message}");
            return default;
        }
    }
}