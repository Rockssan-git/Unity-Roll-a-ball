using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public enum Language { EN, FR };

public class MainManager : MonoBehaviour
{
    public static MainManager Instance; // static = shared by all instances of this class
    public Language CurrentLanguage { get; private set; } = Language.FR;
    

    public float questionnaireFreezeDuration = 1f;
    private int currentLevel = 1;

    [SerializeField] private List<string> levelScenesNames;
    [SerializeField] public bool allowSceneTransition;

    public void LoadNextLevel()
    {
        if (!allowSceneTransition){
            return;
        }

        currentLevel++;
        if (currentLevel-1 < levelScenesNames.Count)
        {
            SceneManager.LoadScene(levelScenesNames[currentLevel-1]);
        }
    }

    

    private void Awake()
    {
        Debug.Log("MANAGER AWAKE");
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
        CurrentLanguage = (Language)PlayerPrefs.GetInt("lang", 0);  
    }

    public void SetLanguage(Language lang)
    {
        CurrentLanguage = lang;
        PlayerPrefs.SetInt("lang", (int)lang); // use to outlive the current session
        PlayerPrefs.Save(); // flush to disk now
    }

    public string LocalizedPath(string baseName)
    {
        string code = CurrentLanguage == Language.FR ? "FR" : "EN";
        return $"{code}/{baseName}";
    }

    public static List<float> CreateRandomList(int nItems, int rangeStart, int rangeEnd)
    {
        List<float> numbers = new List<float>();
        for (int i = 0; i < nItems; i++)
            numbers.Add(Random.Range(rangeStart, rangeEnd+1));   // rangeStart to rangeEnd inclusive
        return numbers;
    }
}
