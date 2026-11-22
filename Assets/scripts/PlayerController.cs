using UnityEngine.SceneManagement;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;   // for BehaviorParameters


// ReSharper disable once CheckNamespace
public class PlayerController : Agent
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


    // Niveau 4 : Saut
    public bool CanJump = false;      // À cocher uniquement dans Level4 et suivants
    public float JumpForce = 7f;      // Puissance du saut
    private bool _isGrounded = true;  // Vérifie si la boule touche le sol
    //rock1
    private Rigidbody _player;

    // Use this for initialization
    private int _score;
    private Vector3 _playerStart;
    private GameObject[] _allPickups;
    private Color _winTextColor;

    // use for automatic agent
    private Vector3[] _allPickupsPositions;
    private readonly Queue<Vector3> _trajectory = new Queue<Vector3>();

    [SerializeField] private int _trajectoryLength = 25;    // number of past positions observed
    [SerializeField] private int _obsStepInterval = 25;     // physics steps between two samples
    private int _stepsSinceLastObs;

    // for recording
    private Vector2 _lastAction;


    private void Awake()
    {
        // reset VectorObservationSize so that it matches the number of pickups of this level
        int nPickups = GameObject.FindGameObjectsWithTag("pickup").Length;
        var behavior = GetComponent<BehaviorParameters>();
        behavior.BrainParameters.VectorObservationSize = 4 * nPickups + 3 * _trajectoryLength + 3;
    }


    //rock2
    // Agent.OnEnable (Initialize) → LazyInitialize → your OnEpisodeBegin → Reset → SpawnPlayerRandomly
    public override void Initialize()
    {
        _player = GetComponent<Rigidbody>();
        _baseSpeed = Speed;
        Log("Player created");
        _allPickups = GameObject.FindGameObjectsWithTag("pickup");

        _allPickupsPositions = new Vector3[_allPickups.Length];
        for (int i = 0; i < _allPickups.Length; i++)
        {
            _allPickupsPositions[i] = _allPickups[i].transform.position;
        }

        _winTextColor = WinText.color;
        HideWinText();

        // On place le joueur au hasard et on cache le trou au début
        SpawnPlayerRandomly();
        if (Hole != null)
        {
            Hole.SetActive(false);
        }

        //
        GetComponent<DecisionRequester>().DecisionPeriod = 25;
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
            ResetGame();
        }

        // Saut avec la touche Espace
        if (CanJump == true && Input.GetKeyDown(KeyCode.Space) && _isGrounded == true)
        {
            _player.linearVelocity = new Vector3(_player.linearVelocity.x, JumpForce, _player.linearVelocity.z);
            _isGrounded = false;
        }

        // Si le joueur tombe dans le vide (sans être tombé dans le trou)
        if (transform.position.y < -10f && _hasFallenInHole == false)
        {
            LoadQuestionnaire();
        }
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Ground"))
        {
            _isGrounded = true;
        }
    }

    //rock3
    private void ResetGame()
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
        ResetSpeed();
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
            Invoke("LoadQuestionnaire", 1.5f); // Attend 1.5 seconde de chute puis charge le niveau suivant

        }
    }
    private void LoadQuestionnaire()
    {
        if (MainManager.Instance.allowSceneTransition)
        {
            SceneManager.LoadScene("GameQuestionnaire");
        }
    }
    //rock4

    private void IncrementScore()
    {
        _score++;
        AddReward(1f);
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
        // if ... for sampling player/ball positions
        _stepsSinceLastObs++;
        if (_stepsSinceLastObs >= _obsStepInterval)
        {
            _stepsSinceLastObs = 0;
            _trajectory.Enqueue(transform.localPosition);
            SessionDataStore.Instance?.RecordStep(StepCount, transform.localPosition, _player.linearVelocity, _lastAction, _score, GetCumulativeReward());
            while (_trajectory.Count > _trajectoryLength)
                _trajectory.Dequeue();                     // keep only the most recent samples
        }
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

    ////////////////////////////////////////////////////////
    /// Automatic agent components
    //////////////////////////////////////////////////////// 


    public override void OnEpisodeBegin()
    {
        SessionDataStore.Instance?.StartEpisode();

        ResetGame();                                   // respawn ball, pickups, score, hole
        _trajectory.Clear();
        for (int i = 0; i < _trajectoryLength; i++)
            _trajectory.Enqueue(transform.localPosition);
        _stepsSinceLastObs = 0;
    }

    // should return list of goals coordinates + trajectory of states
    // trajectory includes current state (last state). 
    // Take this and filter in Python if required to get only current state
    public override void CollectObservations(VectorSensor sensor)
    {
        // Pickups: position + whether still collectable (4 floats each)
        foreach (GameObject pickup in _allPickups)
        {
            if (pickup == null) // destroyed (scene unloading)
            {
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0f);
                continue;
            }

            sensor.AddObservation(pickup.transform.localPosition);
            sensor.AddObservation(pickup.activeSelf ? 1f : 0f);
        }

        // Past positions, oldest first, padded to a fixed length (3 floats each)
        Vector3[] past = _trajectory.ToArray();
        foreach (Vector3 p in past)
            sensor.AddObservation(p);

        // Current position, always last
        sensor.AddObservation(transform.localPosition);
    }

    // Heuristic controller gains (tune in the Inspector)
    [SerializeField] private float _heuristicGain = 1f;      // how strongly to push toward the target
    [SerializeField] private float _heuristicDamping = 0.5f; // how strongly to resist current velocity

    private Transform FindNearestPickup()
    {
        Transform nearest = null;
        float minimumDistance = float.MaxValue;

        foreach (GameObject pickup in _allPickups)
        {
            if (pickup == null || !pickup.activeInHierarchy)
                continue;

            float distance = Vector3.Distance(transform.position, pickup.transform.position);
            if (distance < minimumDistance)
            {
                minimumDistance = distance;
                nearest = pickup.transform;
            }
        }
        return nearest;
    }

    // Nearest remaining pickup, or the hole once every pickup is collected
    private Transform FindTarget()
    {
        Transform pickup = FindNearestPickup();
        if (pickup != null)
            return pickup;
        if (Hole != null && Hole.activeInHierarchy)
            return Hole.transform;
        return null;
    }

    [SerializeField] private bool _manualControl = true;   // keyboard if true, automatic controller if false

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;
        actions[0] = 0f;
        actions[1] = 0f;

        if (_hasFallenInHole || _player == null)
            return;

        if (_manualControl)
        {
            actions[0] = Input.GetAxisRaw(HORIZONTAL_AXIS);   // A/D or ←/→ : -1, 0 or 1
            actions[1] = Input.GetAxisRaw(VERTICAL_AXIS);     // S/W or ↓/↑
            return;
        }

        // Automatic controller (unchanged)
        Transform target = FindTarget();
        if (target == null)
            return;

        Vector3 direction = target.position - transform.position;
        actions[0] = Mathf.Clamp(direction.x * _heuristicGain - _player.linearVelocity.x * _heuristicDamping, -1f, 1f);
        actions[1] = Mathf.Clamp(direction.z * _heuristicGain - _player.linearVelocity.z * _heuristicDamping, -1f, 1f);
        Debug.DrawLine(transform.position, target.position, Color.green);
    }

    // Applies actions from the heuristic or from the trained model
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (_hasFallenInHole)
            return;

        var a = actions.ContinuousActions;
        _lastAction = new Vector2(a[0], a[1]);
        _player.AddForce(new Vector3(a[0], 0f, a[1]) * Speed);

        if (_score == _allPickups.Length & !_manualControl)
        {
            EndEpisode();
        }
    }
}