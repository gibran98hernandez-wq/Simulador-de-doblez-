import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Simulador de doblado simétrico", layout="wide")
st.title("Simulador matemático de doblado simétrico de tubo")
st.caption(
    "El tubo se sujeta en el centro. La receta se introduce desde afuera hacia adentro, "
    "con posiciones absolutas medidas desde el centro hasta el inicio de cada curva. "
    "La rotación del tubo solo se realiza antes de cada doblez y se interpreta como absoluta (0° a 360°)."
)

# ----------------------- Geometría -----------------------

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        raise ValueError("Se encontró un vector de longitud cero.")
    return v / n

def rotate_about_axis(v, axis, angle_rad):
    """Rotación de Rodrigues de v alrededor de axis."""
    axis = unit(axis)
    return (v * np.cos(angle_rad)
            + np.cross(axis, v) * np.sin(angle_rad)
            + axis * np.dot(axis, v) * (1 - np.cos(angle_rad)))

def build_half(recipe, side=1, samples_per_bend=35):
    """
    Construye geométricamente una mitad desde el centro hacia el extremo.
    La receta se ordena por distancia ascendente para construir la forma,
    aunque el usuario la introduce en orden de ejecución exterior -> centro.

    distance_mm: posición absoluta desde el centro hasta el INICIO del arco,
    medida sobre la longitud desarrollada del tubo.
    """
    # Construimos primero la mitad +X.
    p = np.array([0.0, 0.0, 0.0])
    t = np.array([1.0, 0.0, 0.0])  # dirección longitudinal inicial
    n = np.array([0.0, 1.0, 0.0])  # dirección del primer doblado
    points = [p.copy()]
    bend_starts, bend_ends = [], []
    straight_segments = []
    bend_info = []
    developed_position = 0.0

    for idx, row in recipe.iterrows():
        station = float(row["distance_mm"])
        angle_deg = float(row["angle_deg"])
        radius = float(row["radius_mm"])
        rot_before = float(row["rot_before_deg"])

        if station < developed_position - 1e-7:
            raise ValueError(
                f"Operación {idx+1}: la distancia desde el centro ({station:g} mm) "
                f"es menor que la posición desarrollada actual ({developed_position:.2f} mm). "
                "Revisa el orden o las distancias."
            )
        if radius <= 0:
            raise ValueError(f"Operación {idx+1}: el radio debe ser mayor que cero.")
        if angle_deg < 0 or angle_deg >= 180:
            raise ValueError(f"Operación {idx+1}: usa ángulos entre 0° y menos de 180°.")

        # La rotación es ABSOLUTA: se aplica sobre el marco de referencia
        # transportado, sin acumular el valor de la fila anterior.
        if rot_before < 0.0 or rot_before > 360.0:
            raise ValueError(
                f"Operación {idx+1}: la rotación debe estar entre 0° y 360°."
            )
        n_bend = unit(rotate_about_axis(n, t, np.deg2rad(rot_before)))
        n_bend = unit(n_bend - np.dot(n_bend, t) * t)

        # Distancia de alimentación desde el centro hasta el inicio del arco.
        straight_len = station - developed_position
        if straight_len > 1e-8:
            start = p.copy()
            p = p + straight_len * t
            straight_segments.append((start.copy(), p.copy(), idx + 1))
            points.append(p.copy())
            developed_position += straight_len

        bend_start = p.copy()
        bend_starts.append(bend_start.copy())

        # Arco circular en el plano definido por t y n.
        amax = np.deg2rad(angle_deg)
        angles = np.linspace(0, amax, max(8, int(samples_per_bend * angle_deg / 90)))
        arc_points = np.array([
            p + radius * np.sin(a) * t + radius * (1 - np.cos(a)) * n_bend
            for a in angles
        ])
        points.extend(arc_points[1:])
        p = arc_points[-1].copy()

        old_t = t.copy()
        old_n = n_bend.copy()
        t = unit(np.cos(amax) * old_t + np.sin(amax) * old_n)
        n = unit(-np.sin(amax) * old_t + np.cos(amax) * old_n)

        arc_length = radius * amax
        developed_position += arc_length
        bend_ends.append(p.copy())

        bend_info.append({
            "Doblez (construcción centro→extremo)": idx + 1,
            "Distancia desde centro al inicio (mm)": station,
            "Inicio X": bend_start[0], "Inicio Y": bend_start[1], "Inicio Z": bend_start[2],
            "Fin X": p[0], "Fin Y": p[1], "Fin Z": p[2],
            "Longitud arco": arc_length,
            "Distancia desarrollada": developed_position
        })

    # Añadimos un tramo recto de salida para poder visualizar y medir paralelismo.
    return {
        "points": np.asarray(points),
        "straight_segments": straight_segments,
        "bend_info": bend_info,
        "end": p,
        "tangent": t,
        "normal": n,
        "developed": developed_position
    }

def reflect_half(half):
    """Reflejo de la mitad +X respecto al plano YZ para obtener la mitad -X."""
    M = np.diag([-1.0, 1.0, 1.0])
    result = dict(half)
    result["points"] = half["points"] @ M.T
    result["straight_segments"] = [
        (a @ M.T, b @ M.T, k) for a, b, k in half["straight_segments"]
    ]
    result["bend_info"] = half["bend_info"]
    result["end"] = half["end"] @ M.T
    result["tangent"] = half["tangent"] @ M.T
    result["normal"] = half["normal"] @ M.T
    return result

def line_line_distance(a0, a1, b0, b1):
    """Distancia mínima entre dos líneas infinitas; informa si son paralelas."""
    u = unit(a1 - a0)
    v = unit(b1 - b0)
    w = a0 - b0
    cross = np.cross(u, v)
    if np.linalg.norm(cross) < 1e-7:
        delta = b0 - a0
        perp = delta - np.dot(delta, u) * u
        return float(np.linalg.norm(perp)), True
    d = abs(np.dot(w, cross)) / np.linalg.norm(cross)
    return float(d), False

# ----------------------- Interfaz -----------------------

with st.sidebar:
    st.header("Dimensiones generales")
    tube_length = st.number_input(
        "Longitud total del tubo (mm)", min_value=1.0, value=2000.0,
        step=0.1, format="%.3f"
    )
    tube_od = st.number_input(
        "Diámetro exterior (mm)", min_value=0.1, value=25.0,
        step=0.1, format="%.3f"
    )
    show_centerline = st.checkbox("Mostrar trayectoria central", value=True)
    st.markdown("---")
    st.info(
        "La distancia de cada doblez es ABSOLUTA desde el centro hasta donde "
        "empieza la curva. Introduce la receta en orden de ejecución: de afuera "
        "hacia adentro (por ejemplo: 300, 200, 100 mm)."
    )

st.subheader("Receta de doblado")
st.write(
    "Una fila por doblez. Introduce las distancias en el orden en que trabaja la máquina: "
    "desde afuera hacia el centro (distancias descendentes). La rotación indicada se aplica "
    "inmediatamente antes de realizar ese doblez. No existe rotación posterior al doblez."
)

default_recipe = pd.DataFrame([
    {"distance_mm": 300.0, "angle_deg": 45.0, "radius_mm": 100.0,
     "rot_before_deg": 0.0},
    {"distance_mm": 200.0, "angle_deg": 45.0, "radius_mm": 100.0,
     "rot_before_deg": 90.0},
])

edited = st.data_editor(
    default_recipe,
    num_rows="dynamic",
    width="stretch",
    column_config={
        "distance_mm": st.column_config.NumberColumn("Inicio de curva desde centro (mm)", min_value=0.0, step=0.1, format="%.3f"),
        "angle_deg": st.column_config.NumberColumn("Ángulo (°)", min_value=0.0, max_value=179.9, step=0.1, format="%.3f"),
        "radius_mm": st.column_config.NumberColumn("Radio (mm)", min_value=0.1, step=0.1, format="%.3f"),
        "rot_before_deg": st.column_config.NumberColumn("Rotación absoluta antes del doblez (°)", min_value=0.0, max_value=360.0, step=0.1, format="%.3f"),
    }
)

if len(edited) == 0:
    st.warning("Agrega al menos un doblez para generar la simulación.")
    st.stop()

try:
    recipe = edited.reset_index(drop=True).copy()
    for column in ["distance_mm", "angle_deg", "radius_mm", "rot_before_deg"]:
        recipe[column] = pd.to_numeric(recipe[column], errors="raise").astype(float)
    # Las posiciones se capturan en orden exterior -> centro (descendente).
    # Para reconstruir la forma física desde el centro, se procesan en orden ascendente.
    recipe_geometry = recipe.sort_values("distance_mm", ascending=True).reset_index(drop=True)

    half_length = tube_length / 2.0
    if (recipe_geometry["distance_mm"] > half_length).any():
        raise ValueError(
            f"Cada inicio de curva debe estar dentro de la mitad del tubo "
            f"(máximo {half_length:.3f} mm desde el centro)."
        )

    right = build_half(recipe_geometry)
    left = reflect_half(right)

    # Completar cada mitad hasta el extremo físico del tubo.
    remaining_length = half_length - right["developed"]
    if remaining_length < -1e-7:
        raise ValueError(
            f"La receta necesita {right['developed']:.3f} mm desde el centro, "
            f"pero solo hay {half_length:.3f} mm disponibles por lado. "
            "Reduce ángulos/radios o acerca los dobleces al centro."
        )
    remaining_length = max(0.0, remaining_length)

    right_tip = right["end"] + remaining_length * right["tangent"]
    left_tip = left["end"] + remaining_length * left["tangent"]
    right_points = np.vstack([right["points"], right_tip])
    left_points = np.vstack([left["points"], left_tip])

    # Agregar los tramos rectos finales para poder medirlos.
    right["straight_segments"].append((right["end"].copy(), right_tip.copy(), "Salida"))
    left["straight_segments"].append((left["end"].copy(), left_tip.copy(), "Salida"))

    # Figura 3D
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=right_points[:, 0], y=right_points[:, 1], z=right_points[:, 2],
        mode="lines", line=dict(width=10), name="Mitad +X"
    ))
    fig.add_trace(go.Scatter3d(
        x=left_points[:, 0], y=left_points[:, 1], z=left_points[:, 2],
        mode="lines", line=dict(width=10), name="Mitad -X"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=6, color="black"), name="Centro de sujeción"
    ))

    for i, (x, y, z) in enumerate(right["points"]):
        if i == 0:
            continue
    for j, info in enumerate(right["bend_info"]):
        fig.add_trace(go.Scatter3d(
            x=[info["Inicio X"], info["Fin X"]],
            y=[info["Inicio Y"], info["Fin Y"]],
            z=[info["Inicio Z"], info["Fin Z"]],
            mode="markers+text",
            marker=dict(size=4),
            text=[f"B{j+1} inicio", f"B{j+1} fin"],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=25, b=0),
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data"
        ),
        legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Resultados")
    col1, col2, col3 = st.columns(3)
    col1.metric("Doblez programados", str(len(recipe)))
    col2.metric("Longitud disponible por mitad", f"{half_length:.2f} mm")
    col3.metric("Longitud restante recta por mitad", f"{remaining_length:.2f} mm")

    # Separación de tramos rectos correspondientes, cuando existan.
    st.markdown("**Separación entre tramos rectos correspondientes**")
    distances = []
    right_straights = right["straight_segments"]
    left_straights = left["straight_segments"]
    for (ra, rb, idx), (la, lb, _) in zip(right_straights, left_straights):
        d, parallel = line_line_distance(ra, rb, la, lb)
        distances.append({
            "Tramo antes del doblez": f"Doblez {idx}",
            "Separación entre ejes (mm)": d,
            "Paralelos": "Sí" if parallel else "No"
        })
    if distances:
        st.dataframe(pd.DataFrame(distances), width="stretch", hide_index=True)
    else:
        st.write("No hay tramos rectos suficientes para calcular la separación.")

    st.markdown("**Coordenadas de inicio y fin de cada arco (mitad +X)**")
    bend_df = pd.DataFrame(right["bend_info"])
    st.dataframe(bend_df.round(3), width="stretch", hide_index=True)

    csv = bend_df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar coordenadas CSV", csv, "coordenadas_dobleces.csv", "text/csv")

    st.warning(
        "IMPORTANTE: modelo geométrico preliminar. La longitud total se divide en dos "
        "mitades iguales. Las distancias se interpretan como posiciones absolutas desde el centro "
        "hasta el inicio del arco, sobre la longitud desarrollada. La receta se captura en orden "
        "exterior→centro, pero la geometría se reconstruye centro→extremo. No incluye springback, "
        "deformación del material, compensaciones de máquina, offsets de sujeción ni colisiones. "
        "Valida con una pieza conocida antes de usar resultados para producción."
    )

except Exception as e:
    st.error(f"No se pudo generar la geometría: {e}")
    st.info("Verifica que las distancias estén dentro de la mitad de la longitud total, que los radios sean positivos y que la longitud desarrollada quepa en cada mitad.")
