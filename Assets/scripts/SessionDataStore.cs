using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

// ---------- Rows of the two tables ----------

[Serializable]
public class TrajectoryRow
{
    public int episode_id;
    public int step_id;              // agent step within the episode (physics steps)
    public float time;               // seconds since the session started
    public float pos_x, pos_y, pos_z;
    public float vel_x, vel_z;
    public float action_x, action_z; // last action applied
    public int score;
    public float cumulative_reward;
}

[Serializable]
public class QuestionnaireRow
{
    public string questionnaire_id;
    public int question_index;       // order in which the question was shown
    public string question;
    public string answer;
    public string timestamp;         // ISO 8601, local time
}

// What is sent to Python
[Serializable]
public class SessionData
{
    public string session_id;
    public string output_dir;
    public List<TrajectoryRow> trajectories = new List<TrajectoryRow>();
    public List<QuestionnaireRow> answers = new List<QuestionnaireRow>();
}

// What Python sends back
[Serializable]
public class PythonSaveResult
{
    public bool ok;
    public int n_trajectory_rows;
    public int n_answer_rows;
    public string trajectories_path;
    public string answers_path;
}

/// <summary>
/// Session-wide store for trajectories and questionnaire answers.
/// Survives scene changes; sends everything to Python with SendToPython().
/// Put it on a root GameObject (together with a PythonRunner) in the first scene.
/// </summary>
public class SessionDataStore : MonoBehaviour
{
    public static SessionDataStore Instance { get; private set; }

    [Tooltip("Script that saves the data, relative to Assets/StreamingAssets.")]
    [SerializeField] private string saveScriptRelativePath = "Python/session_store.py";

    [Tooltip("Leave empty to use the PythonRunner on this GameObject.")]
    [SerializeField] private PythonRunner pythonRunner;

    [Tooltip("Folder for the CSV files. Empty = Application.persistentDataPath.")]
    [SerializeField] private string outputDirectory = "Resources/StoredData";

    public int CurrentEpisodeId { get; private set; } = -1;

    private SessionData _data;
    private float _sessionStartTime;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);   // keep the store created by the first scene
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        if (pythonRunner == null)
            pythonRunner = GetComponent<PythonRunner>();

        _sessionStartTime = Time.time;
        _data = new SessionData
        {
            session_id = DateTime.Now.ToString("yyyyMMdd_HHmmss"),
            output_dir = !string.IsNullOrEmpty(outputDirectory) ? outputDirectory
                : Application.isEditor ? Path.GetFullPath(Path.Combine(Application.dataPath, "..", "StoredData"))
                : Application.persistentDataPath,
        };
    }

    public void StartEpisode()
    {
        CurrentEpisodeId++;
    }

    public void RecordStep(int stepId, Vector3 position, Vector3 velocity, Vector2 action,
                           int score, float cumulativeReward)
    {
        if (CurrentEpisodeId < 0)
            StartEpisode();

        _data.trajectories.Add(new TrajectoryRow
        {
            episode_id = CurrentEpisodeId,
            step_id = stepId,
            time = Time.time - _sessionStartTime,
            pos_x = position.x, pos_y = position.y, pos_z = position.z,
            vel_x = velocity.x, vel_z = velocity.z,
            action_x = action.x, action_z = action.y,
            score = score,
            cumulative_reward = cumulativeReward,
        });
    }

    public void RecordAnswer(string questionnaireId, int questionIndex, string question, string answer)
    {
        _data.answers.Add(new QuestionnaireRow
        {
            questionnaire_id = questionnaireId,
            question_index = questionIndex,
            question = question,
            answer = answer,
            timestamp = DateTime.Now.ToString("o"),
        });
    }

    /// <summary>
    /// Sends all rows recorded since the last successful send to Python, which appends
    /// them to the session's CSV files. Blocks until Python is done.
    /// </summary>
    public PythonSaveResult SendToPython(bool clearAfterSuccess = true)
    {
        if (pythonRunner == null)
        {
            Debug.LogError("[SessionDataStore] No PythonRunner assigned.");
            return null;
        }

        PythonSaveResult result = pythonRunner.RunPython<SessionData, PythonSaveResult>(_data, saveScriptRelativePath);

        if (result != null && result.ok)
        {
            Debug.Log($"[SessionDataStore] Saved {result.n_trajectory_rows} trajectory rows to {result.trajectories_path} " +
                      $"and {result.n_answer_rows} answers to {result.answers_path}");
            if (clearAfterSuccess)
            {
                _data.trajectories.Clear();
                _data.answers.Clear();
            }
        }
        else
        {
            Debug.LogError("[SessionDataStore] Saving failed; data kept in memory for the next attempt.");
        }
        return result;
    }
}