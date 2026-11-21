using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    public GameObject ObstaclePrefab;   // Le prefab de l'obstacle
    public float SpawnInterval = 3f;    // Un obstacle toutes les 3 secondes
    public float SpawnRange = 7f;       // Zone d'apparition sur le plateau
    public int MaxObstacles = 8;        // Limite pour ne pas bloquer tout le plateau

    private float _timer = 0f;
    private int _currentCount = 0;

    private void Update()
    {
        if (_currentCount >= MaxObstacles)
        {
            return;
        }

        _timer = _timer + Time.deltaTime;

        if (_timer >= SpawnInterval)
        {
            SpawnObstacle();
            _timer = 0f;
        }
    }

    private void SpawnObstacle()
    {
        float randomX = Random.Range(-SpawnRange, SpawnRange);
        float randomZ = Random.Range(-SpawnRange, SpawnRange);
        Vector3 spawnPosition = new Vector3(randomX, 0.5f, randomZ);

        Instantiate(ObstaclePrefab, spawnPosition, Quaternion.identity);
        _currentCount++;
    }
}