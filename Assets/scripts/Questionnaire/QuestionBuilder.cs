using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class QuestionBuilder : MonoBehaviour
{
    [SerializeField] private bool shuffleQuestions;

    [SerializeField] protected TextMeshProUGUI questionLabel;
    [SerializeField] private string questionsFilePath;
    [SerializeField] private string optionsFilePath;

    [SerializeField] private Toggle togglePrefab;
    protected ToggleGroup toggleGroup;
    protected readonly List<Toggle> toggles = new List<Toggle>();

    protected List<string> questions;
    protected List<string> options;
    protected int currentIndex = 0;

    // for saving
    [SerializeField] private string questionnaireID = "";

    // GameObject to deactivate at the end
    [SerializeField] private List<GameObject> objectsToDeactivateOnFinish;


    void Start()
    {
        toggleGroup = GetComponent<ToggleGroup>();
        toggleGroup.allowSwitchOff = true;

        questions = LoadQuestions();

        if (shuffleQuestions){
            // Fisher-Yates shuffle: walk from the end, swap each element with a
            // random earlier one. Unbiased and in-place.
            for (int i = questions.Count - 1; i > 0; i--)
            {
                int j = Random.Range(0, i + 1);        // 0..i inclusive
                (questions[i], questions[j]) = (questions[j], questions[i]);   // swap
            }
        }

        options = LoadLines(MainManager.Instance.LocalizedPath(optionsFilePath));

        if (questions == null || options == null) return;   // error already logged
        if (questions.Count == 0)
        {
            Debug.LogError("[Questionnaire] Questions file is empty.");
            return;
        }
        if (options.Count == 0)
        {
            Debug.LogError("[Questionnaire] Options file is empty.");
            return;
        }

        BuildToggles();
        ShowQuestion(0);
    }

    private List<string> LoadLines(string path)
    {
        TextAsset loaded = Resources.Load<TextAsset>(path);
        if (loaded == null)
        {
            Debug.LogError($"[Questionnaire] File not found: {path}");
            return null;
        }
        return loaded.text
            .Split('\n')
            .Select(line => line.Trim())
            .Where(line => line.Length > 0)
            .ToList();
    }

    private void BuildToggles()
    {
        for (int i = 0; i < options.Count; i++)
        {
            Toggle t = Instantiate(togglePrefab, transform, false);
            t.SetIsOnWithoutNotify(false);
            t.group = toggleGroup;

            var label = t.GetComponentInChildren<TextMeshProUGUI>();
            if (label != null) label.text = options[i];

            int choiceIndex = i;                        // capture per iteration
            t.onValueChanged.AddListener(isOn =>
            {
                if (isOn) { StartCoroutine(OnAnswered(options[choiceIndex])); }
            });

            toggles.Add(t);
        }
    }

    protected virtual void ShowQuestion(int index)
    {
        questionLabel.text = questions[index];
        ResetToggles();
    }

    protected virtual List<string> LoadQuestions()
    {
        return LoadLines(MainManager.Instance.LocalizedPath(questionsFilePath));
    }

    private void ResetToggles()
    {
        foreach (var t in toggles)
            t.SetIsOnWithoutNotify(false);
    }

    private IEnumerator OnAnswered(string choiceText)
    {
        // Record before advancing
        ExperimentDataHandler.Instance?.SaveQuestionnaireResponse(questionnaireID, questions[currentIndex], choiceText);
        SessionDataStore.Instance?.RecordAnswer(questionnaireID, currentIndex, questions[currentIndex], choiceText);

        // pause then advance
        yield return StartCoroutine(Pause(MainManager.Instance.questionnaireFreezeDuration));

        currentIndex++;
        if (currentIndex < questions.Count)
            ShowQuestion(currentIndex);
        else
            OnFinished();
    }

    public IEnumerator Pause(float duration)
    {
        Time.timeScale = 0f;
        SetTogglesInteractable(false);
        yield return new WaitForSecondsRealtime(duration);
        SetTogglesInteractable(true);
        Time.timeScale = 1f;
    }

    private void SetTogglesInteractable(bool state)
    {
        foreach (var t in toggles)
            t.interactable = state;
    }

    private void OnFinished()
    {
        foreach (var t in toggles) t.gameObject.SetActive(false);
        foreach (var go in objectsToDeactivateOnFinish) go.SetActive(false);

        if (MainManager.Instance.CurrentLanguage == Language.EN)
        {
            questionLabel.text = "Thank you!";
            MainManager.Instance.LoadNextLevel();
        }
        else
        {
            questionLabel.text = "Merci !";
            MainManager.Instance.LoadNextLevel();
        }
        Debug.Log("[Questionnaire] Complete.");

        // Save
        ExperimentDataHandler.Instance?.EndSessionBatch();
        SessionDataStore.Instance?.SendToPython();
    }
}