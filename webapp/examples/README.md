# Registros de ejemplo

Siete registros de ECG de 12 derivaciones (10 s, 500 Hz) tomados del **conjunto de test** del
PhysioNet/Computing in Cardiology Challenge 2020. Permiten probar la demo sin datos propios:
seis con predicción acertada del ensemble y uno con diferencias entre predicción y etiqueta
(para mostrar también los límites del modelo).

Se generan con `python tools/build_examples.py --predictions runs/eval/ensemble-f1-platt/predictions_test.csv --dataset dataset2020`
(selección reproducible, `--seed 0`). Si solo cambia la calibración o los umbrales, `--refresh-references` (sin `--dataset`) actualiza las probabilidades de referencia de los mismos siete registros. `examples.json` guarda además las probabilidades de referencia de
la evaluación, con las que `tests/test_examples.py` comprueba la app.

## Licencia y cita

Datos: *Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020*,
licencia **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Al usarlos, citar:

- Perez Alday, E. A., Gu, A., Shah, A., Liu, C., Sharma, A., Seyedi, S., Bahrami Rad, A., Reyna, M., & Clifford, G. (2022).
  *Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020* (version 1.0.2). PhysioNet.
  https://doi.org/10.13026/dvyd-kd57
- Perez Alday E. A. et al. Physiol Meas. 2020. http://doi.org/10.1088/1361-6579/abc960

Los archivos `.hea`/`.mat` se incluyen sin modificar. Origen de cada registro en `examples.json` (`source_db`).
