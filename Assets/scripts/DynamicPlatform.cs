using UnityEngine;

public class DynamicPlatform : MonoBehaviour
{
    // Déplacement (se rapprocher / s'éloigner)
    public Vector3 MoveDirection = new Vector3(1f, 0f, 0f); // Direction du mouvement (ex: axe X)
    public float MoveDistance = 2f;                         // Distance max de déplacement
    public float MoveSpeed = 1.5f;                          // Vitesse du mouvement

    // Taille (s'agrandir / se réduire)
    public bool EnableScaling = true;
    public float ScaleAmount = 0.2f;                        // Variation de taille
    public float ScaleSpeed = 1.5f;                         // Vitesse de variation

    private Vector3 _startPosition;
    private Vector3 _startScale;

    private void Start()
    {
        _startPosition = transform.position;
        _startScale = transform.localScale;
    }

    private void Update()
    {
        // Mouvement fluide d'aller-retour avec Sinus
        float moveOffset = Mathf.Sin(Time.time * MoveSpeed) * MoveDistance;
        transform.position = _startPosition + (MoveDirection * moveOffset);

        // Agrandissement et réduction fluides
        if (EnableScaling == true)
        {
            float scaleOffset = Mathf.Sin(Time.time * ScaleSpeed) * ScaleAmount;
            float newX = _startScale.x + scaleOffset;
            float newZ = _startScale.z + scaleOffset;
            transform.localScale = new Vector3(newX, _startScale.y, newZ);
        }
    }

}