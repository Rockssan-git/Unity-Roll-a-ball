using UnityEngine;

public class SpeedBallSpawner : MonoBehaviour
{
    public GameObject SpeedUpPrefab;    // Prefab boule verte (+)
    public GameObject SpeedDownPrefab;  // Prefab boule bleue (-)
    public float SpawnInterval = 4f;    // Une boule toutes les 4 secondes
    public float SpawnRange = 7f;       // Zone d'apparition sur le plateau

    private float _timer = 0f;

    private void Update()
    {
        _timer = _timer + Time.deltaTime;

        if (_timer >= SpawnInterval)
        {
            SpawnRandomBall();
            _timer = 0f;
        }
    }

    private void SpawnRandomBall()
    {
        float randomX = Random.Range(-SpawnRange, SpawnRange);
        float randomZ = Random.Range(-SpawnRange, SpawnRange);
        Vector3 spawnPosition = new Vector3(randomX, 0.5f, randomZ);

        // 1 chance sur 2 de faire apparaître une boule (+) ou (-)
        int choice = Random.Range(0, 2);

        if (choice == 0)
        {
            Instantiate(SpeedUpPrefab, spawnPosition, Quaternion.identity);
        }
        else
        {
            Instantiate(SpeedDownPrefab, spawnPosition, Quaternion.identity);
        }
    }
}