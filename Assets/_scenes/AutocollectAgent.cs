
using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AutocollectAgent : Agent
{
    [Header("Agent")]
    public Transform pickupsParent;
    public float speed = 10f;
    public float acceleration = 1.5f;
    public float braking = 1.2f;
    public bool logActions = false;

    [Header("Dashboard")]
    public bool showDashboard = true;
    public float sampleInterval = 0.5f;

    private Rigidbody rb;
    private int actionSteps;

    // Metriques
    private int totalPickups;
    private int collected;
    private int wallCollisions;
    private int previousRemaining;

    private float startTime;
    private float finalElapsed;
    private float lastPickupTime;
    private float distanceTraveled;
    private float maximumSpeed;
    private float nextSampleTime;

    private Vector3 lastPosition;
    private bool completed;

    private struct Sample
    {
        public float time;
        public float velocity;
        public int pickups;
        public int collisions;
    }

    private readonly List<Sample> history =
        new List<Sample>();

    private readonly List<float> pickupTimes =
        new List<float>();

    private Vector2 scrollPosition;

    // Temps fige lorsque la mission est terminee
    private float ElapsedTime
    {
        get
        {
            return completed
                ? finalElapsed
                : Mathf.Max(0f, Time.time - startTime);
        }
    }

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();

        if (rb == null || pickupsParent == null)
        {
            Debug.LogError(
                "AutocollectAgent : Rigidbody ou Pickups Parent manquant."
            );

            enabled = false;
            return;
        }

        ResetMetrics();

        Debug.Log(
            "AutocollectAgent et dashboard initialises !"
        );
    }

    private void ResetMetrics()
    {
        totalPickups = CountActivePickups();
        previousRemaining = totalPickups;

        collected = 0;
        wallCollisions = 0;
        actionSteps = 0;

        completed = false;

        startTime = Time.time;
        finalElapsed = 0f;
        lastPickupTime = 0f;

        distanceTraveled = 0f;
        maximumSpeed = 0f;

        lastPosition = transform.position;

        nextSampleTime = Time.time + 0.5f;

        history.Clear();
        pickupTimes.Clear();

        RecordSample();
    }

    private int CountActivePickups()
    {
        if (pickupsParent == null)
            return 0;

        int count = 0;

        foreach (Transform pickup in pickupsParent)
        {
            if (pickup != null &&
                pickup.gameObject.activeInHierarchy)
            {
                count++;
            }
        }

        return count;
    }

    private Transform FindNearestPickup()
    {
        if (pickupsParent == null)
            return null;

        Transform nearest = null;
        float minimumDistance = float.MaxValue;

        foreach (Transform pickup in pickupsParent)
        {
            if (pickup == null ||
                !pickup.gameObject.activeInHierarchy)
                continue;

            float distance = Vector3.Distance(
                transform.position,
                pickup.position
            );

            if (distance < minimumDistance)
            {
                minimumDistance = distance;
                nearest = pickup;
            }
        }

        return nearest;
    }

    public override void CollectObservations(
        VectorSensor sensor)
    {
        Transform target = FindNearestPickup();

        Vector3 direction = target != null
            ? target.position - transform.position
            : Vector3.zero;

        sensor.AddObservation(direction.x);
        sensor.AddObservation(direction.z);

        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }

    public override void OnActionReceived(
        ActionBuffers actions)
    {
        // STOP : aucune nouvelle force apres la mission
        if (completed || rb == null)
            return;

        float horizontal = actions.ContinuousActions[0];
        float vertical = actions.ContinuousActions[1];

        Vector3 movement = new Vector3(
            horizontal, 0f, vertical
        );

        rb.AddForce(movement * speed);

        actionSteps++;

        if (logActions && actionSteps % 40 == 0)
        {
            Debug.Log(
                $"Agent Action : X={horizontal:F2}, Z={vertical:F2}"
            );
        }
    }

    public override void Heuristic(
        in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;

        actions[0] = 0f;
        actions[1] = 0f;

        if (completed || rb == null)
            return;

        Transform target = FindNearestPickup();

        if (target == null)
            return;

        Vector3 direction =
            target.position - transform.position;

        actions[0] = Mathf.Clamp(
            direction.x * acceleration
            - rb.linearVelocity.x * braking,
            -1f, 1f
        );

        actions[1] = Mathf.Clamp(
            direction.z * acceleration
            - rb.linearVelocity.z * braking,
            -1f, 1f
        );

        Debug.DrawLine(
            transform.position,
            target.position,
            Color.green
        );
    }

    private void Update()
    {
        if (rb == null || pickupsParent == null)
            return;

        // IMPORTANT : arreter toutes les mesures
        // lorsque la mission est terminee
        if (completed)
            return;

        float elapsed = Time.time - startTime;

        Vector3 currentPosition = transform.position;

        // Distance parcourue sur le plateau
        distanceTraveled += Vector2.Distance(
            new Vector2(lastPosition.x, lastPosition.z),
            new Vector2(currentPosition.x, currentPosition.z)
        );

        lastPosition = currentPosition;

        float currentSpeed =
            rb.linearVelocity.magnitude;

        maximumSpeed = Mathf.Max(
            maximumSpeed,
            currentSpeed
        );

        int remaining = CountActivePickups();

        if (remaining < previousRemaining)
        {
            int newCollections =
                previousRemaining - remaining;

            for (int i = 0; i < newCollections; i++)
            {
                collected++;

                float duration = Mathf.Max(
                    0f,
                    elapsed - lastPickupTime
                );

                pickupTimes.Add(duration);
                lastPickupTime = elapsed;
            }

            Debug.Log(
                $"COLLECTE : {collected}/{totalPickups} | " +
                $"Temps : {elapsed:F2}s"
            );

            // TERMINER LA MISSION
            if (totalPickups > 0 &&
                collected >= totalPickups)
            {
                CompleteMission(elapsed);
                return;
            }
        }

        previousRemaining = remaining;

        if (Time.time >= nextSampleTime)
        {
            RecordSample();

            nextSampleTime = Time.time +
                Mathf.Max(0.1f, sampleInterval);
        }
    }

    // NOUVEAU : figer les resultats finaux
    private void CompleteMission(float elapsed)
    {
        completed = true;

        // Sauvegarder le temps final
        finalElapsed = elapsed;

        // Arreter la boule
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Dernier point des graphiques
        RecordSample();

        float finalEfficiency =
            finalElapsed > 0.001f
            ? collected * 60f / finalElapsed
            : 0f;

        Debug.Log(
            "========== MISSION TERMINEE ==========\n" +
            $"Cubes collectes : {collected}/{totalPickups}\n" +
            $"Temps final : {finalElapsed:F2} s\n" +
            $"Efficacite finale : {finalEfficiency:F2} cubes/min\n" +
            $"Distance parcourue : {distanceTraveled:F2} u\n" +
            $"Collisions avec murs : {wallCollisions}\n" +
            "======================================"
        );
    }

    private void RecordSample()
    {
        Sample sample = new Sample
        {
            time = ElapsedTime,
            velocity = rb != null
                ? rb.linearVelocity.magnitude : 0f,
            pickups = collected,
            collisions = wallCollisions
        };

        history.Add(sample);

        if (history.Count > 1200)
            history.RemoveAt(0);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (completed)
            return;

        Transform current = collision.transform;

        while (current != null)
        {
            if (
                current.name.Equals(
                    "Walls",
                    StringComparison.OrdinalIgnoreCase
                ) ||
                current.name.StartsWith(
                    "Wall",
                    StringComparison.OrdinalIgnoreCase
                )
            )
            {
                wallCollisions++;
                return;
            }

            current = current.parent;
        }
    }

    private float GetGraphValue(Sample sample, int type)
    {
        switch (type)
        {
            case 0: return sample.pickups;
            case 1: return sample.velocity;
            case 2: return sample.collisions;
            default: return 0f;
        }
    }

    private void DrawLine(
        Vector2 a,
        Vector2 b,
        Color color,
        float thickness = 2f)
    {
        Vector2 direction = b - a;

        if (direction.sqrMagnitude < 0.001f)
            return;

        Matrix4x4 previousMatrix = GUI.matrix;
        Color previousColor = GUI.color;

        GUI.color = color;

        float angle = Mathf.Atan2(
            direction.y,
            direction.x
        ) * Mathf.Rad2Deg;

        GUIUtility.RotateAroundPivot(angle, a);

        GUI.DrawTexture(
            new Rect(
                a.x,
                a.y - thickness * 0.5f,
                direction.magnitude,
                thickness
            ),
            Texture2D.whiteTexture
        );

        GUI.matrix = previousMatrix;
        GUI.color = previousColor;
    }

    private void DrawLineGraph(
        float y,
        float width,
        string title,
        int type,
        float maximum,
        Color color)
    {
        GUI.Label(
            new Rect(8, y, width - 16, 20),
            title
        );

        Rect graph = new Rect(
            8, y + 22, width - 20, 76
        );

        GUI.Box(graph, GUIContent.none);

        float left = graph.x + 29f;
        float right = graph.xMax - 10f;
        float top = graph.y + 9f;
        float bottom = graph.yMax - 20f;

        Color axisColor = new Color(
            0.55f, 0.55f, 0.55f
        );

        DrawLine(
            new Vector2(left, top),
            new Vector2(left, bottom),
            axisColor, 1f
        );

        DrawLine(
            new Vector2(left, bottom),
            new Vector2(right, bottom),
            axisColor, 1f
        );

        maximum = Mathf.Max(1f, maximum);

        GUI.Label(
            new Rect(graph.x + 2, top - 5, 27, 18),
            maximum.ToString("F0")
        );

        GUI.Label(
            new Rect(graph.x + 12, bottom - 10, 18, 18),
            "0"
        );

        if (history.Count < 2)
            return;

        float start = history[0].time;
        float end = history[history.Count - 1].time;

        float duration = Mathf.Max(0.1f, end - start);

        int stride = Mathf.Max(
            1,
            history.Count / 90
        );

        Vector2 previous = Vector2.zero;
        bool hasPoint = false;

        for (int i = 0; i < history.Count; i += stride)
        {
            Sample sample = history[i];

            float t = Mathf.Clamp01(
                (sample.time - start) / duration
            );

            float v = Mathf.Clamp01(
                GetGraphValue(sample, type) / maximum
            );

            Vector2 point = new Vector2(
                Mathf.Lerp(left, right, t),
                Mathf.Lerp(bottom, top, v)
            );

            if (hasPoint)
                DrawLine(previous, point, color);

            previous = point;
            hasPoint = true;
        }

        Sample last = history[history.Count - 1];

        Vector2 finalPoint = new Vector2(
            right,
            Mathf.Lerp(
                bottom,
                top,
                Mathf.Clamp01(
                    GetGraphValue(last, type) / maximum
                )
            )
        );

        if (hasPoint)
            DrawLine(previous, finalPoint, color);

        GUI.Label(
            new Rect(left, bottom + 2, 65, 16),
            start.ToString("F0") + "s"
        );

        GUI.Label(
            new Rect(right - 57, bottom + 2, 57, 16),
            end.ToString("F0") + "s"
        );
    }

    private void DrawPickupTimes(float y, float width)
    {
        GUI.Label(
            new Rect(8, y, width - 16, 20),
            "4. Temps par cube (secondes)"
        );

        Rect graph = new Rect(
            8, y + 22, width - 20, 83
        );

        GUI.Box(graph, GUIContent.none);

        if (pickupTimes.Count == 0)
        {
            GUI.Label(
                new Rect(
                    graph.x + 10,
                    graph.y + 25,
                    graph.width - 20,
                    22
                ),
                "Aucun cube collecte pour le moment"
            );

            return;
        }

        float maximum = 1f;

        foreach (float duration in pickupTimes)
        {
            maximum = Mathf.Max(maximum, duration);
        }

        int columns = Mathf.Max(
            totalPickups,
            pickupTimes.Count
        );

        float chartWidth = graph.width - 42f;
        float columnWidth =
            chartWidth / Mathf.Max(1, columns);

        float baseY = graph.yMax - 20f;
        float maxBarHeight = graph.height - 34f;

        GUI.Label(
            new Rect(graph.x + 2, graph.y + 2, 35, 18),
            maximum.ToString("F1")
        );

        for (int i = 0; i < pickupTimes.Count; i++)
        {
            float barHeight =
                pickupTimes[i] / maximum * maxBarHeight;

            float x = graph.x + 32f +
                i * columnWidth;

            Color oldColor = GUI.color;

            GUI.color = new Color(1f, 0.72f, 0.2f);

            GUI.DrawTexture(
                new Rect(
                    x + 1,
                    baseY - barHeight,
                    Mathf.Max(2f, columnWidth - 3f),
                    barHeight
                ),
                Texture2D.whiteTexture
            );

            GUI.color = oldColor;

            if (columnWidth >= 14f)
            {
                GUI.Label(
                    new Rect(
                        x,
                        baseY + 1,
                        columnWidth,
                        17
                    ),
                    (i + 1).ToString()
                );
            }
        }
    }

    private void OnGUI()
    {
        if (rb == null)
            return;

        if (!showDashboard)
        {
            if (GUI.Button(
                new Rect(
                    Screen.width - 145,
                    8,
                    135,
                    28
                ),
                "Afficher stats"))
            {
                showDashboard = true;
            }

            return;
        }

        float panelWidth = Mathf.Min(
            370f,
            Mathf.Max(250f, Screen.width * 0.37f)
        );

        float panelHeight = Mathf.Min(
            630f,
            Screen.height - 16f
        );

        float panelX =
            Screen.width - panelWidth - 8f;

        GUI.Box(
            new Rect(
                panelX,
                8,
                panelWidth,
                panelHeight
            ),
            ""
        );

        GUI.Label(
            new Rect(
                panelX + 12,
                14,
                panelWidth - 115,
                24
            ),
            completed
                ? "MISSION TERMINEE"
                : "ROLL-A-BALL | AGENT STATS"
        );

        if (GUI.Button(
            new Rect(
                panelX + panelWidth - 97,
                12,
                85,
                25
            ),
            "Masquer"))
        {
            showDashboard = false;
        }

        // Temps fige en fin de partie
        float elapsed = ElapsedTime;

        float currentSpeed = completed
            ? 0f
            : rb.linearVelocity.magnitude;

        // Efficacite figee en fin de partie
        float efficiency = elapsed > 0.001f
            ? collected * 60f / elapsed
            : 0f;

        float lastDuration = pickupTimes.Count > 0
            ? pickupTimes[pickupTimes.Count - 1]
            : 0f;

        float contentWidth = panelWidth - 38f;

        scrollPosition = GUI.BeginScrollView(
            new Rect(
                panelX + 9,
                45,
                panelWidth - 18,
                panelHeight - 48
            ),
            scrollPosition,
            new Rect(0, 0, contentWidth, 710)
        );

        GUI.Label(
            new Rect(8, 8, contentWidth - 10, 20),
            $"Cubes collectes : {collected}/{totalPickups}"
        );

        GUI.Label(
            new Rect(8, 30, contentWidth - 10, 20),
            completed
                ? $"Temps final : {elapsed:F2} s"
                : $"Temps ecoule : {elapsed:F1} s"
        );

        GUI.Label(
            new Rect(8, 52, contentWidth - 10, 20),
            $"Vitesse : {currentSpeed:F2} u/s"
        );

        GUI.Label(
            new Rect(8, 74, contentWidth - 10, 20),
            $"Distance parcourue : {distanceTraveled:F2} u"
        );

        GUI.Label(
            new Rect(8, 96, contentWidth - 10, 20),
            completed
                ? $"Efficacite finale : {efficiency:F2} cubes/min"
                : $"Efficacite : {efficiency:F2} cubes/min"
        );

        GUI.Label(
            new Rect(8, 118, contentWidth - 10, 20),
            $"Collisions avec murs : {wallCollisions}"
        );

        GUI.Label(
            new Rect(8, 140, contentWidth - 10, 20),
            $"Dernier cube : {lastDuration:F1} s"
        );

        DrawLineGraph(
            174,
            contentWidth,
            "1. Cubes collectes / temps",
            0,
            totalPickups,
            Color.green
        );

        DrawLineGraph(
            292,
            contentWidth,
            "2. Vitesse / temps",
            1,
            maximumSpeed,
            Color.cyan
        );

        DrawLineGraph(
            410,
            contentWidth,
            "3. Collisions cumulees / temps",
            2,
            wallCollisions,
            new Color(1f, 0.4f, 0.4f)
        );

        DrawPickupTimes(
            528,
            contentWidth
        );

        if (completed)
        {
            GUI.Label(
                new Rect(
                    8,
                    665,
                    contentWidth - 12,
                    25
                ),
                "MISSION TERMINEE : 100% des cubes !"
            );
        }

        GUI.EndScrollView();
    }
}
