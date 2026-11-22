
using System.Collections.Generic;
using UnityEngine;

public class PickupCalibrationManager : MonoBehaviour
{
    [Header("References")]
    public Transform pickupsParent;
    public PlayerController playerController;

    [Header("Calibration")]
    public int triggerAfterCollected = 4;
    public int pickupsToHide = 16;

    private List<GameObject> allPickups = new List<GameObject>();
    private HashSet<GameObject> hiddenPickups =
        new HashSet<GameObject>();

    private int initialCount;
    private bool reductionApplied = false;
    private bool holeActivated = false;

    void Start()
    {
        if (pickupsParent == null || playerController == null)
        {
            Debug.LogError(
                "Calibration : references manquantes !"
            );
            enabled = false;
            return;
        }

        foreach (Transform child in pickupsParent)
        {
            if (child.CompareTag("pickup"))
            {
                allPickups.Add(child.gameObject);
            }
        }

        initialCount = allPickups.Count;

        Debug.Log(
            $"CALIBRATION : {initialCount} pickups detectes."
        );
    }

    private int CountActivePickups()
    {
        int count = 0;

        foreach (GameObject pickup in allPickups)
        {
            if (pickup != null && pickup.activeInHierarchy)
                count++;
        }

        return count;
    }

    // Ne pas compter les cubes caches comme collectes
    private int CountPlayerCollections()
    {
        int count = 0;

        foreach (GameObject pickup in allPickups)
        {
            if (pickup != null &&
                !pickup.activeInHierarchy &&
                !hiddenPickups.Contains(pickup))
            {
                count++;
            }
        }

        return count;
    }

    private void LateUpdate()
    {
        if (initialCount == 0)
            return;

        // Detecter une reinitialisation du niveau
        if (reductionApplied &&
            CountActivePickups() == initialCount)
        {
            hiddenPickups.Clear();
            reductionApplied = false;
            holeActivated = false;

            Debug.Log("CALIBRATION : niveau reinitialise.");
        }

        int collected = CountPlayerCollections();

        // Declencher la disparition apres 4 collectes
        if (!reductionApplied &&
            collected >= triggerAfterCollected)
        {
            HideRemainingPickups();
        }

        collected = CountPlayerCollections();

        int requiredCollections =
            initialCount - hiddenPickups.Count;

        // Affichage du score reel
        if (playerController.ScoreText != null)
        {
            playerController.ScoreText.text =
                $"Score: {collected}/{requiredCollections}";
        }

        // Tous les cubes necessaires ont ete collectes
        if (!holeActivated &&
            CountActivePickups() == 0 &&
            collected >= requiredCollections)
        {
            ActivateHole();
        }
    }

    private void HideRemainingPickups()
    {
        reductionApplied = true;

        List<GameObject> available =
            new List<GameObject>();

        foreach (GameObject pickup in allPickups)
        {
            if (pickup != null && pickup.activeInHierarchy)
            {
                available.Add(pickup);
            }
        }

        int amount = Mathf.Min(
            pickupsToHide,
            available.Count
        );

        // Selection aleatoire de 16 cubes
        for (int i = 0; i < amount; i++)
        {
            int index = Random.Range(0, available.Count);

            GameObject selected = available[index];

            hiddenPickups.Add(selected);
            selected.SetActive(false);

            available.RemoveAt(index);
        }

        Debug.Log(
            $"CALIBRATION : {amount} cubes disparus ! " +
            $"{available.Count} cubes restent."
        );
    }

    private void ActivateHole()
    {
        if (playerController.Hole == null)
        {
            Debug.LogWarning(
                "CALIBRATION : aucun Hole configure."
            );
            return;
        }

        GameObject hole = playerController.Hole;

        float range = playerController.SpawnRange;

        float x = Random.Range(-range, range);
        float z = Random.Range(-range, range);

        hole.transform.position =
            new Vector3(x, 0.01f, z);

        hole.SetActive(true);
        holeActivated = true;

        Debug.Log(
            "CALIBRATION TERMINEE : le trou apparait !"
        );
    }
}
