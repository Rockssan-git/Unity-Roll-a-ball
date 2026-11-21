using UnityEngine.SceneManagement;
using System;
using UnityEngine;
using UnityEngine.UI;

// ReSharper disable once CheckNamespace
public class PlayerController : MonoBehaviour
{
    private const string VERTICAL_AXIS = "Vertical";
    private const string HORIZONTAL_AXIS = "Horizontal";
    public Text ScoreText;
    public Text WinText;
    public float Speed = 5F;
    public float TableFriction = 5F;
    public float FrictionRampSpeed = 1F;
    public float FrictionRamp = 2.5F;
    //rock1
    // Variables pour le Niveau 1
    private float _baseSpeed;
    public GameObject Hole;            // Le trou
    public Collider GroundCollider;    // Le sol du plateau
    public float SpawnRange = 7f;      // Zone de spawn aléatoire
    private bool _hasFallenInHole = false;
    //rock1
    private Rigidbody _player;

    // Use this for initialization
    private int _score;
    private Vector3 _playerStart;
    private GameObject[] _allPickups;
    private Color _winTextColor;

    // discretize time action
    [SerializeField] private int _actionStepInterval = 100;   // 100 × 0.02 s = 2 s
    private int _stepsSinceLastAction;



    //rock2
    private void Start()
    {
        _player = GetComponent<Rigidbody>();
        _baseSpeed = Speed;
        Log("Player created");
        _allPickups = GameObject.FindGameObjectsWithTag("pickup");
        _winTextColor = WinText.color;
        HideWinText();

        // On place le joueur au hasard et on cache le trou au début
        SpawnPlayerRandomly();
        if (Hole != null)
        {
            Hole.SetActive(false);
        }
    }

    private void SpawnPlayerRandomly()
    {
        float randomX = UnityEngine.Random.Range(-SpawnRange, SpawnRange);
        float randomZ = UnityEngine.Random.Range(-SpawnRange, SpawnRange);

        _playerStart = new Vector3(randomX, 0.5f, randomZ);
        _player.position = _playerStart;
        _player.linearVelocity = Vector3.zero;
    }
    //rock2
    private void HideWinText()
    {
        WinText.color = new Color(0, 0, 0, 0);
    }

    private void ShowWinText()
    {
        WinText.color = _winTextColor;
    }

    // Update is called once per frame
    private void Update()
    {
        if (Input.GetKey(KeyCode.R))
        {
            Reset();
        }
    }

    //rock3
    private void Reset()
    {
        _score = 0;
        _hasFallenInHole = false;

        // On réactive le sol et on recache le trou
        if (GroundCollider != null)
        {
            GroundCollider.enabled = true;
        }
        if (Hole != null)
        {
            Hole.SetActive(false);
        }

        SpawnPlayerRandomly();
        foreach (GameObject pickup in _allPickups)
        {
            pickup.SetActive(true);
        }
        UpdateScoreDisplay();
        HideWinText();
    }
    //rock3

    //rock4
    void OnTriggerEnter(Collider other)
    {
        GameObject hitObject = other.gameObject;
        Log("Colliding with: {0}", hitObject.tag);

        if (hitObject.CompareTag("pickup"))
        {
            IncrementScore();
            hitObject.SetActive(false);
        }
        else if (hitObject.CompareTag("Hole") && _hasFallenInHole == false)
        {
            _hasFallenInHole = true;

            // On désactive le sol pour laisser tomber la boule
            if (GroundCollider != null)
            {
                GroundCollider.enabled = false;
            }
            ShowWinText();
            Invoke("LoadNextLevel", 1.5f); // Attend 1.5 seconde de chute puis charge le niveau suivant

}
    }
    private void LoadNextLevel()
    {
        int nextSceneIndex = SceneManager.GetActiveScene().buildIndex + 1;
        if (nextSceneIndex < SceneManager.sceneCountInBuildSettings)
        {
            SceneManager.LoadScene(nextSceneIndex);
        }
    }
    //rock4

    private void IncrementScore()
    {
        _score++;
        UpdateScoreDisplay();
    }

    //rock5
    private void UpdateScoreDisplay()
    {
        if (ScoreText != null)
        {
            ScoreText.text = "Score: " + _score;
            if (_score == _allPickups.Length)
            {
                SpawnHole();
            }
        }
    }

    private void SpawnHole()
    {
        if (Hole != null && Hole.activeSelf == false)
        {
            float randomX = UnityEngine.Random.Range(-SpawnRange, SpawnRange);
            float randomZ = UnityEngine.Random.Range(-SpawnRange, SpawnRange);

            Hole.transform.position = new Vector3(randomX, 0.01f, randomZ);
            Hole.SetActive(true);
            Log("Le trou est apparu !");
        }
    }
    //rock5

    // also called once per frame but guaranteed to be called last
    // also called once per frame but guaranteed to be called last
    public void FixedUpdate()
    {
        UpdatePlayerForceFromInput();
    }

    private void UpdatePlayerForceFromInput()
    {
        // GetAxisRaw renvoie -1, 0 ou 1 instantanément (sans délai)
        float moveX = Input.GetAxisRaw(HORIZONTAL_AXIS);
        float moveZ = Input.GetAxisRaw(VERTICAL_AXIS);

        // Si le joueur appuie ou maintient une touche de déplacement
        if (moveX != 0 || moveZ != 0)
        {
            Vector3 direction = new Vector3(moveX, 0, moveZ).normalized;// On déplace la boule tout en gardant sa vitesse Y (pour qu'elle puisse tomber dans le trou)
            _player.linearVelocity = new Vector3(direction.x * Speed, _player.linearVelocity.y, direction.z * Speed);
            _stepsSinceLastAction = 0;
        }
        else
        {
            // Si le joueur relâche les touches : on décélère rapidement jusqu'à l'arrêt
            float vx = _player.linearVelocity.x * 0.8f;
            float vz = _player.linearVelocity.z * 0.8f;

            // Dès que la vitesse devient très faible, on l'arrête net
            if (Mathf.Abs(vx) < 0.1f)
            {
                vx = 0f;
            }
            if (Mathf.Abs(vz) < 0.1f)
            {
                vz = 0f;
            }

            _player.linearVelocity = new Vector3(vx, _player.linearVelocity.y, vz);
        }
    }

    private void ApplyTableSurfaceFriction()
    {
        var velocity = _player.linearVelocity;
        var decelX = GetDecellerationFor(velocity.x);
        var decelZ = GetDecellerationFor(velocity.z);
        _player.AddForce(new Vector3(decelX, 0, decelZ));
    }

    private void Log(string message, params object[] args)
    {
        Debug.Log(string.Format(message, args));
    }

    private float GetDecellerationFor(float f)
    {
        var absoluteForce = Math.Abs(f);
        var situationMultiplier = absoluteForce < FrictionRampSpeed ? FrictionRamp : 1;   // the slower you're going, the more friction of the table matters
        var direction = f < 0 ? 1 : -1; // should oppose the current movement
        var max = (float)(TableFriction * direction * situationMultiplier * 0.1);
        return Math.Abs(max) < absoluteForce ? -1 * absoluteForce : max;
    }
    // Niveau 3 : modifier temporairement la vitesse du joueur
    public void ChangeSpeed(float multiplier, float duration)
    {
        Speed = _baseSpeed * multiplier;
        CancelInvoke("ResetSpeed");
        Invoke("ResetSpeed", duration);
    }

    private void ResetSpeed()
    {
        Speed = _baseSpeed;
    }
}