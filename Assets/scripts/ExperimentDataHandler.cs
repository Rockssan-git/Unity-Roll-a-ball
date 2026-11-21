using System;
using System.Text;
using System.Globalization;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

public class ExperimentDataHandler : MonoBehaviour
{
    public static ExperimentDataHandler Instance;

    private string serverUrl = "https://railway-generalizationgridtask-production.up.railway.app";
    private string apiKey;

    private string sessionId;
    private string prolificPid = "";
    private string studyId = "";
    private string prolificSessionId = "";


    // Keep every row in memory so we can re-send the full set at the end
    // as a safety net against an individual dropped request.
    private List<string> sessionBatchRows = new List<string>();

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        var keyAsset = Resources.Load<TextAsset>("railway_post_key");
        apiKey = keyAsset != null ? keyAsset.text.Trim() : "";

        // Read Prolific parameters from the launch URL (WebGL only).
        ParseUrlParams();

        string stamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string guid = Guid.NewGuid().ToString("N").Substring(0, 8);
        // Prefix the session id with the PID when present, so it's easy to match.
        sessionId = string.IsNullOrEmpty(prolificPid)
            ? $"{guid}_{stamp}"
            : $"{prolificPid}_{guid}_{stamp}";
    }

    public bool IsOnProlific()
    {
        return !string.IsNullOrEmpty(prolificPid);
    }

    private void ParseUrlParams()
    {
        string url = Application.absoluteURL;
        if (string.IsNullOrEmpty(url) || !url.Contains("?")) return;

        string query = url.Substring(url.IndexOf('?') + 1);
        foreach (string pair in query.Split('&'))
        {
            int eq = pair.IndexOf('=');
            if (eq < 0) continue;
            string key = pair.Substring(0, eq);
            string val = Uri.UnescapeDataString(pair.Substring(eq + 1));

            // match whatever key names you put in the Prolific study URL
            if (key == "participant" || key == "PROLIFIC_PID") prolificPid = val;
            else if (key == "study_id" || key == "STUDY_ID")   studyId = val;
            else if (key == "session"  || key == "SESSION_ID") prolificSessionId = val;
        }
    }


    // Exploration task (the grid). Keeps the existing call-site signature.
    public void SaveTrialSelectedCell(int trialID, Vector3 cellPosition, int rowID, int columnID, float guessValue, float cellValue)
    {
        var data = new Dictionary<string, object>
        {
            { "data_type",         "exploration" },
            { "trial",             trialID },
            { "position_x",        cellPosition.x },
            { "position_y",        cellPosition.y },
            { "position_rowID",    rowID },
            { "position_columnID", columnID },
            { "predicted_value",   guessValue },
            { "actual_value",      cellValue },
        };
        Save(data);
    }

    // Per-participant questionnaire.
    public void SaveQuestionnaireResponse(string questionnaireID, string question, string response)
    {
        var data = new Dictionary<string, object>
        {
            { "data_type",    "questionnaire" },
            { "questionnaire_id",    questionnaireID },
            { "question",  question },
            { "response",  response },
        };
        Save(data);
    }

    // Self-reference paradigm. Matches the fields the question builder logs.
    public void SaveSelfReferenceResponse(string subphase, int itemIndex,
        string item, int choiceIndex, string choiceText)
    {
        var data = new Dictionary<string, object>
        {
            { "data_type",    "self_reference" },
            { "subphase",     subphase },       // "1", "2a", "2b"
            { "item_index",   itemIndex },
            { "item",        item },
            { "choice_index", choiceIndex },
            { "choice_text",  choiceText },
        };
        Save(data);
    }

    // utils
    public void SaveClick(string object_id)
    {
        var data = new Dictionary<string, object>
        {
            { "data_type",    "click" },
            { "object_id",    object_id },
        };
        Save(data);
    }

    public void SaveException(string exceptionName)
    {
        var data = new Dictionary<string, object>
        {
            { "data_type",    "exception" },
            { "exception_id",    exceptionName },
        };
        Save(data);
    }

    // ---- Shared send path ----------------------------------------------

    // Adds the fields common to every row, then buffers + posts.
    private void Save(Dictionary<string, object> data)
    {
        data["session_id"] = sessionId;
        data["language"] = MainManager.Instance != null ? MainManager.Instance.CurrentLanguage.ToString() : "unknown";
        data["context"] = string.IsNullOrEmpty(prolificPid) ? "in_person" : "prolific";
        data["prolific_pid"] = prolificPid;          // empty for in-person
        data["study_id"] = studyId;
        data["prolific_session"] = prolificSessionId;

        DateTime now = DateTime.UtcNow;
        data["client_time"] = now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ",
                                                    CultureInfo.InvariantCulture);

        string json = ToJson(data);
        sessionBatchRows.Add(json); // for the end-of-session batch
    }

    // Call once when the experiment ends. Re-sends the complete dataset as one
    // batch, so even if a per-row POST failed mid-session, nothing is lost.
    public void EndSessionBatch()
    {
        string batch = "[" + string.Join(",", sessionBatchRows) + "]";
        StartCoroutine(Post(batch));
        sessionBatchRows = new List<string>();
    }

    private IEnumerator Post(string jsonPayload)
    {
        byte[] body = Encoding.UTF8.GetBytes(jsonPayload);

        using (var req = new UnityWebRequest($"{serverUrl}/data", "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("X-API-Key", apiKey);
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[Data] Upload failed ({sessionId}): {req.error}");
            else
                Debug.Log($"[Data] Saved ({sessionId})");
        }
    }

    // ---- Locale-safe JSON builder --------------------------------------
    private string ToJson(Dictionary<string, object> data)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var kv in data)
        {
            if (!first) sb.Append(",");
            first = false;
            sb.Append("\"").Append(kv.Key).Append("\":").Append(Format(kv.Value));
        }
        return sb.Append("}").ToString();
    }

    private string Format(object v)
    {
        switch (v)
        {
            case null: return "null";
            case bool b: return b ? "true" : "false";
            case float f: return f.ToString(CultureInfo.InvariantCulture);
            case double d: return d.ToString(CultureInfo.InvariantCulture);
            case int i: return i.ToString(CultureInfo.InvariantCulture);
            default:
                string s = v.ToString().Replace("\\", "\\\\").Replace("\"", "\\\"");
                return "\"" + s + "\"";
        }
    }
}