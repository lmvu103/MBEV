import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression


# --- Fluid Property Functions ---

def calculate_Bw(p, t):
    """Calculate water formation volume factor using McCain's correlation."""
    delta_VwP = (-1.95301e-9 * p * t - 1.72834e-13 * (p ** 2) * t - 3.58922e-7 * p - 2.25341e-10 * (p ** 2))
    delta_VwT = (-1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * (t ** 2))
    return (1 + delta_VwP) * (1 + delta_VwT)


def compute_z(Sg, p_val, t):
    """Compute Z-factor using Newton-Raphson method."""
    T_pc = 120.1 + 425 * Sg - 62.9 * (Sg ** 2)
    P_pc = 671.1 + 14 * Sg - 34.3 * (Sg ** 2)
    T_pr = (t + 460) / T_pc
    P_pr = p_val / P_pc
    Z = 1.0
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        rho_r = 0.27 * P_pr / (Z * T_pr)
        term1 = (0.3265 - 1.07 / T_pr - 0.5339 / (T_pr ** 3) + 0.01569 / (T_pr ** 4) - 0.05165 / (T_pr ** 5)) * rho_r
        term2 = (0.5475 - 0.7361 / T_pr + 0.1844 / (T_pr ** 2)) * (rho_r ** 2)
        term3 = -0.1056 * ((-0.7361 / T_pr) + (0.1844 / (T_pr ** 2))) * (rho_r ** 5)
        term4 = 0.6134 * (1 + 0.7210 * (rho_r ** 2)) * ((rho_r ** 2) / (T_pr ** 3)) * np.exp(-0.7210 * (rho_r ** 2))
        F_val = 1 + term1 + term2 + term3 + term4
        f_Z = Z - F_val
        delta = 1e-6
        Z_plus = Z + delta
        rho_r_plus = 0.27 * P_pr / (Z_plus * T_pr)
        term1_plus = (0.3265 - 1.07 / T_pr - 0.5339 / (T_pr ** 3) + 0.01569 / (T_pr ** 4) - 0.05165 / (
                T_pr ** 5)) * rho_r_plus
        term2_plus = (0.5475 - 0.7361 / T_pr + 0.1844 / (T_pr ** 2)) * (rho_r_plus ** 2)
        term3_plus = -0.1056 * ((-0.7361 / T_pr) + (0.1844 / (T_pr ** 2))) * (rho_r_plus ** 5)
        term4_plus = 0.6134 * (1 + 0.7210 * (rho_r_plus ** 2)) * ((rho_r_plus ** 2) / (T_pr ** 3)) * np.exp(
            -0.7210 * (rho_r_plus ** 2))
        F_plus = 1 + term1_plus + term2_plus + term3_plus + term4_plus
        f_Z_plus = Z_plus - F_plus
        Z_minus = Z - delta
        rho_r_minus = 0.27 * P_pr / (Z_minus * T_pr)
        term1_minus = (0.3265 - 1.07 / T_pr - 0.5339 / (T_pr ** 3) + 0.01569 / (T_pr ** 4) - 0.05165 / (
                T_pr ** 5)) * rho_r_minus
        term2_minus = (0.5475 - 0.7361 / T_pr + 0.1844 / (T_pr ** 2)) * (rho_r_minus ** 2)
        term3_minus = -0.1056 * ((-0.7361 / T_pr) + (0.1844 / (T_pr ** 2))) * (rho_r_minus ** 5)
        term4_minus = 0.6134 * (1 + 0.7210 * (rho_r_minus ** 2)) * ((rho_r_minus ** 2) / (T_pr ** 3)) * np.exp(
            -0.7210 * (rho_r_minus ** 2))
        F_minus = 1 + term1_minus + term2_minus + term3_minus + term4_minus
        f_Z_minus = Z_minus - F_minus
        derivative = (f_Z_plus - f_Z_minus) / (2 * delta)
        if abs(derivative) < 1e-12:
            break
        Z_new = Z - f_Z / derivative
        if abs(Z_new - Z) < tol:
            Z = Z_new
            break
        Z = Z_new
    return Z


def calculate_gas_viscosity(p, t, sg_g, Z):
    """Calculate gas viscosity using Lee et al. correlation."""
    M = 28.97 * sg_g
    rho_g = (28.97 * sg_g * p) / (Z * 10.73 * (t + 460))
    K = ((9.4 + 0.02 * M) * ((t + 460) ** 1.5)) / (209 + 19 * M + t + 460)
    X = 3.5 + (986 / (t + 460)) + 0.01 * M
    Y = 2.4 - 0.2 * X
    return 1e-4 * K * np.exp(X * ((rho_g / 62.4) ** Y))


# Oil-specific functions
def calculate_Bo(p, p_b, r_s_input, sg_g, api, t):
    sg_o = 141.5 / (api + 131.5)
    if p <= p_b:
        Rs_p = calculate_Rs(p, p_b, r_s_input, sg_g, api, t)
        return 0.9759 + 0.00012 * (((Rs_p * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
    else:
        rs_bp = sg_g * ((p_b * (10 ** (0.0125 * api))) / (18 * (10 ** (0.00091 * t)))) ** 1.2048
        bo_b = 0.9759 + 0.00012 * (((rs_bp * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
        c_o = ((5 * rs_bp) + (17.2 * t) - (1180 * sg_g) + (12.61 * api) - 1433) / ((p + 14.7) * 100000)
        return bo_b * np.exp(-c_o * (p - p_b))


def calculate_Rs(p, p_b, r_s_input, sg_g, api, t):
    if p > p_b:
        return r_s_input
    else:
        return sg_g * (((p + 14.7) * (10 ** (0.0125 * api)) / (18 * (10 ** (0.00091 * t)))) ** 1.2048)


def calculate_Bg(p, sg_g, t):
    Z_val = compute_z(sg_g, p, t)
    return 0.005035 * Z_val * (t + 460) / p


def calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g):
    v_od = 10 ** (10 ** (3.0324 - 0.02023 * api) * t ** (-1.163)) - 1
    if p <= p_b:
        Rs_val = calculate_Rs(p, p_b, r_s_input, sg_g, api, t)
        return (10.715 * (Rs_val + 100) ** (-0.515)) * v_od ** (5.44 * (Rs_val + 150) ** (-0.338))
    else:
        rs_bp = sg_g * ((p_b * (10 ** (0.0125 * api))) / (18 * (10 ** (0.00091 * t)))) ** 1.2048
        v_o_sat = (10.715 * (rs_bp + 100) ** (-0.515)) * v_od ** (5.44 * (r_s_input + 150) ** (-0.338))
        return v_o_sat + 0.001 * (p - p_b) * ((0.024 * v_o_sat ** 1.6) + (0.038 * v_o_sat ** 0.56))


# --- Main Application ---

def main():
    st.title("**Material Balance Simulator**")
    reservoir_type = st.selectbox("Select Reservoir Type", ["Oil", "Gas"], key="reservoir_type")

    if reservoir_type == "Oil":

        # Data source selection and file handling
        data_source = st.selectbox("1.Select Data Source", ["Upload File", "Use Example Data"], key="data_source")

        if data_source == "Upload File":
            uploaded_file = st.file_uploader("2.Upload Production Data (Excel)", type=["xls", "xlsx"], )
            st.write(">>*NOTE*>> Pleas update The excel file should have column names: Date, Pressure, "
                     "Cum Oil Production (MMSTB), Cum Gas Production (MMSCF), Cum Water Production (MMSTB).")
            if uploaded_file:
                history_data = pd.read_excel(uploaded_file)
            else:
                history_data = None
        else:
            try:
                history_data = pd.read_excel("example_oil_data.xlsx")
                st.write(f"Using example data for {reservoir_type} reservoir.")
            except FileNotFoundError:
                st.error(f"Example data file for {reservoir_type} not found.")
                history_data = None

        if history_data is not None:
            required_columns = (
                ['Date', 'Pressure', 'Cum Oil Production', 'Cum Gas Production', 'Cum Water Production']
                if reservoir_type == "Oil"
                else ['Date', 'Pressure', 'Cum Gas Production']
            )
            missing_columns = [col for col in required_columns if col not in history_data.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                # Data inputs

                st.sidebar.header("Reservoir Parameters")
                res_p = st.sidebar.number_input("Reservoir Pressure (psi)", min_value=0.0, value=1500.0, step=50.0)
                t = st.sidebar.number_input("Temperature (F)", min_value=0.0, value=200.0, step=5.0)
                sg_g = st.sidebar.number_input("Gas Specific Gravity", min_value=0.0, value=0.75, step=0.05)
                c_f = st.sidebar.number_input("Rock Compressibility (1/psi)", min_value=0.0, value=6e-6, format="%.2e",
                                              step=0.5e-6)
                c_w = st.sidebar.number_input("Water Compressibility (1/psi)", min_value=0.0, value=1e-6, format="%.2e",
                                              step=0.5e-6)
                S_wc = st.sidebar.number_input("Initial Water Saturation (fraction)", min_value=0.0, max_value=1.0,
                                               value=0.15, step=0.05)
                api = st.sidebar.number_input("API Gravity", min_value=0.0, value=20.0, step=5.0)
                r_s_input = st.sidebar.number_input("Solution Gas-Oil Ratio (SCF/STB)", min_value=0.0, value=200.0,
                                                    step=50.0)
                m = st.sidebar.number_input("Gas Cap Ratio (m)", min_value=0.0, value=0.1, step=0.1)
                dates = pd.to_datetime(history_data['Date'])
                pressure_data = history_data['Pressure'].values
                Np_data = history_data['Cum Oil Production'].values  # MMSTB
                Gp_data = history_data['Cum Gas Production'].values  # MMSCF
                Wp_data = history_data['Cum Water Production'].values  # MMSTB
                n_points = len(pressure_data)
                # Fluid Property Calculations
                p_b = 18.2 * (((r_s_input / sg_g) ** 0.83) * (10 ** (0.00091 * t - 0.0125 * api)) - 1.4)
                st.write(f"Calculated Bubble Point Pressure: {p_b:.2f} psi")
                Rs_data = np.array([calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Bo_data = np.array([calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                Bg_data = np.array([calculate_Bg(p, sg_g, t) for p in pressure_data])
                Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                Viscosity_data = np.array(
                    [calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in pressure_data])
                B_oi, R_si, B_gi = Bo_data[0], Rs_data[0], Bg_data[0]
                #else:
                #    Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                #    Bg_data = 0.02827 * Z_prod * (t + 460) / pressure_data
                #    Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                #    Viscosity_data = np.array(
                #        [calculate_gas_viscosity(p, t, sg_g, z) for p, z in zip(pressure_data, Z_prod)])
                #    B_gi = Bg_data[0]

                # Step 3: Option to Plot Reservoir Plots
                if st.checkbox("Plot Reservoir Production Data"):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.plot(dates, Np_data, 'b-', label='Cum Oil (MMSTB)')
                    ax1.set_xlabel('Time');
                    ax1.set_ylabel('Cum Oil (MMSTB)')
                    ax1.legend();
                    ax1.grid()
                    ax1_twin = ax1.twiny()
                    ax1_twin.plot(pressure_data, Np_data, 'r--', alpha=0)
                    ax1_twin.set_xlabel('Pressure (psi)');
                    ax1_twin.invert_xaxis()

                    ax2.plot(dates, Gp_data, 'g-', label='Cum Gas (MMSCF)')
                    ax2.set_xlabel('Time');
                    ax2.set_ylabel('Cum Gas (MMSCF)')
                    ax2.legend();
                    ax2.grid()
                    ax2_twin = ax2.twiny()
                    ax2_twin.plot(pressure_data, Gp_data, 'r--', alpha=0)
                    ax2_twin.set_xlabel('Pressure (psi)');
                    ax2_twin.invert_xaxis()

                    ax3.plot(dates, Wp_data, 'r-', label='Cum Water (MMSTB)')
                    ax3.set_xlabel('Time');
                    ax3.set_ylabel('Cum Water (MMSTB)')
                    ax3.legend();
                    ax3.grid()
                    ax3_twin = ax3.twiny()
                    ax3_twin.plot(pressure_data, Wp_data, 'r--', alpha=0)
                    ax3_twin.set_xlabel('Pressure (psi)');
                    ax3_twin.invert_xaxis()
                    #else:
                    #    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    #    ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (BSCF)')
                    #    ax1.set_xlabel('Time');
                    #    ax1.set_ylabel('Cum Gas (BSCF)')
                    #    ax1.legend();
                    #    ax1.grid()
                    #    ax1_twin = ax1.twiny()
                    #    ax1_twin.plot(pressure_data, Gp_data, 'r--', alpha=0)
                    #    ax1_twin.set_xlabel('Pressure (psi)');
                    #    ax1_twin.invert_xaxis()
                    #
                    #    ax2.plot(dates, Wp_data, 'g-', label='Cum Water (MMSTB)')
                    #    ax2.set_xlabel('Time');
                    #    ax2.set_ylabel('Cum Water (MMSTB)')
                    #    ax2.legend();
                    #    ax2.grid()
                    #    ax2_twin = ax2.twiny()
                    #    ax2_twin.plot(pressure_data, Wp_data, 'r--', alpha=0)
                    #    ax2_twin.set_xlabel('Pressure (psi)');
                    #    ax2_twin.invert_xaxis()
                    plt.tight_layout()
                    st.pyplot(fig)

                if st.checkbox("Plot Fluid Properties"):
                    plots = (
                        [("Rs (SCF/STB)", Rs_data), ("Bo (RB/STB)", Bo_data), ("Bg (RB/SCF)", Bg_data),
                         ("Bw (RB/STB)", Bw_data), ("Viscosity (cP)", Viscosity_data), ("Z-factor", Z_prod)]
                        if reservoir_type == "Oil"
                        else [("Bg (RB/SCF)", Bg_data), ("Bw (RB/STB)", Bw_data), ("Viscosity (cp)", Viscosity_data),
                              ("Z-factor", Z_prod)]
                    )
                    fig, axs = plt.subplots((len(plots) + 1) // 2, 2, figsize=(12, 6 * ((len(plots) + 1) // 2)))
                    axs = axs.flatten()
                    for ax, (label, data) in zip(axs, plots):
                        ax.plot(pressure_data, data, 'b-', label=label)
                        if reservoir_type == "Oil":
                            ax.axvline(p_b, color='r', linestyle='--', label=f'Pb: {p_b:.2f} psi')
                        ax.set_xlabel('Pressure (psi)');
                        ax.set_ylabel(label)
                        ax.legend();
                        ax.grid()
                    if len(plots) % 2:
                        fig.delaxes(axs[-1])
                    plt.tight_layout()
                    st.pyplot(fig)

                # Step 4: Campbell/Cole Plots
                if st.checkbox("Plot Campbell Plot"):
                    F_data = np.zeros(n_points)
                    for i in range(n_points):
                        gas_term = (Gp_data[i] - Np_data[i] * Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else 0
                        F_data[i] = Np_data[i] * Bo_data[i] + gas_term + Wp_data[i] * Bw_data[i]
                    c_e = (c_w * S_wc + c_f) / (1 - S_wc)
                    E_o_data = np.array(
                        [(Bo_data[i] - B_oi) + (R_si - Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else Bo_data[
                                                                                                                   i] - B_oi
                         for i in range(n_points)])
                    E_g_data = B_oi * (Bg_data / B_gi - 1)
                    E_fw_data = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                    E_t_data = E_o_data + m * E_g_data + E_fw_data
                    F_over_Et = np.where(np.abs(E_t_data) > 1e-6, F_data / E_t_data, np.nan)
                    fig, ax = plt.subplots()
                    ax.plot(F_data, F_over_Et, 'bo-', label='F / E_t vs F')
                    ax.set_xlabel('Cumulative Reservoir Voidage (F)');
                    ax.set_ylabel('F / E_t')
                    ax.set_title('Campbell Plot');
                    ax.legend();
                    ax.grid()
                    st.pyplot(fig)
                else:
                    cole_y = np.full(n_points, np.nan)
                    for i in range(1, n_points):
                        if Bg_data[i] != B_gi:
                            cole_y[i] = Gp_data[i] * Bg_data[i] / (Bg_data[i] - B_gi)
                    fig, ax = plt.subplots()
                    ax.plot(Gp_data[1:], cole_y[1:], 'bo-', label='Cole Plot')
                    ax.set_xlabel('Cumulative Gas (BSCF)');
                    ax.set_ylabel(r'$\frac{G_p\,Bg}{Bg - B_{gi}}$')
                    ax.set_title('Cole Plot');
                    ax.legend();
                    ax.grid()
                    st.pyplot(fig)

                # Step 5: Water Influx
                influx_present = st.selectbox("Is water influx present?", ["No", "Yes"]) == "Yes"
                if influx_present:
                    W_ei = st.number_input("Aquifer Volume (MMft^3)", min_value=0.0, value=0.0)
                    c_t = c_w + c_f
                    We_data = (W_ei / 5.615) * c_t * (res_p - pressure_data)
                    if st.checkbox("Plot Water Influx"):
                        fig, ax = plt.subplots()
                        ax.plot(dates, We_data, 'r--', label='Water Influx (MMBBL)')
                        ax.set_xlabel('Time');
                        ax.set_ylabel('Water Influx (MMBBL)')
                        ax.set_title('Water Influx Over Time');
                        ax.legend();
                        ax.grid()
                        st.pyplot(fig)
                else:
                    We_data = np.zeros(n_points)

                # Step 6: Material Balance Calculations
                if st.checkbox("Calculate Material Balance"):
                    if reservoir_type == "Oil":
                        F_data = np.zeros(n_points)
                        for i in range(n_points):
                            gas_term = (Gp_data[i] - Np_data[i] * Rs_data[i]) * Bg_data[i] if pressure_data[
                                                                                                  i] < p_b else 0
                            F_data[i] = Np_data[i] * Bo_data[i] + gas_term + Wp_data[i] * Bw_data[i]
                            c_e = (c_w * S_wc + c_f) / (1 - S_wc)
                            E_o_data = np.array(
                                [(Bo_data[i] - B_oi) + (R_si - Rs_data[i]) * Bg_data[i] if pressure_data[
                                                                                               i] < p_b else
                                 Bo_data[i] - B_oi for i in range(n_points)])
                            E_g_data = B_oi * (Bg_data / B_gi - 1)
                            E_fw_data = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                            E_t_data = E_o_data + m * E_g_data + E_fw_data
                        if not influx_present:
                            N_intercept = np.sum(E_t_data * F_data) / np.sum(E_t_data ** 2)
                            fig, ax = plt.subplots()
                            ax.plot(E_t_data, F_data, 'bo', label='F vs E_t')
                            ax.plot(E_t_data, N_intercept * E_t_data, 'r-', label=f'N = {N_intercept * 1e6:.2f} STB')
                            ax.set_xlabel('E_t (RB/STB)');
                            ax.set_ylabel('F')
                            ax.set_title('Material Balance (No Influx)');
                            ax.legend();
                            ax.grid()
                            st.pyplot(fig)
                            st.write(f"OOIP (No Influx): {N_intercept * 1e6:.2f} STB")
                        else:
                            F_minus_We = F_data - We_data / 1e6
                            N_intercept = np.sum(E_t_data * F_minus_We) / np.sum(E_t_data ** 2)
                            fig, ax = plt.subplots()
                            ax.plot(E_t_data, F_minus_We, 'bo', label='F - W_e vs E_t')
                            ax.plot(E_t_data, N_intercept * E_t_data, 'r-', label=f'N = {N_intercept * 1e6:.2f} STB')
                            ax.set_xlabel('E_t (RB/STB)');
                            ax.set_ylabel('F - W_e')
                            ax.set_title('Material Balance (With Influx)');
                            ax.legend();
                            ax.grid()
                            st.pyplot(fig)
                            st.write(f"OOIP (With Influx): {N_intercept * 1e6:.2f} STB")
                            # Step 7: Drive Mechanism Analysis
                            if st.checkbox("Plot Drive Mechanism Analysis"):
                                if reservoir_type == "Oil":
                                    E_o_drive = np.array(
                                        [(Bo_data[i] - B_oi) + (r_s_input - Rs_data[i]) * Bg_data[i] if pressure_data[
                                                                                                            i] < p_b else
                                         Bo_data[i] - B_oi for i in range(n_points)])
                                    E_g_drive = B_oi * (Bg_data / B_gi - 1)
                                    E_fw_drive = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                                    We_term = We_data / N_intercept if influx_present else np.zeros(n_points)
                                    total_expansion = E_o_drive + m * E_g_drive + E_fw_drive + We_term
                                    oil_gas_pct = np.nan_to_num((E_o_drive / total_expansion) * 100)
                                    gas_cap_pct = np.nan_to_num((m * E_g_drive / total_expansion) * 100)
                                    rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion) * 100)
                                    water_influx_pct = np.nan_to_num((We_term / total_expansion) * 100)
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.stackplot(pressure_data[1:], oil_gas_pct[1:], gas_cap_pct[1:],
                                                 rock_water_pct[1:],
                                                 water_influx_pct[1:],
                                                 labels=['Oil & Gas', 'Gas Cap', 'Rock & Water', 'Water Influx'],
                                                 colors=['#FF9671', '#00C4B4', '#845EC2', '#FFC75F'])
                                    ax.set_xlabel('Pressure (psi)');
                                    ax.set_ylabel('Contribution (%)')
                                    ax.set_title('Drive Mechanism Analysis');
                                    ax.legend();
                                    ax.grid()
                                    ax.invert_xaxis()
                                    st.pyplot(fig)
                                else:
                                    E_g_drive = Bg_data - B_gi
                                    C_val = (c_f + c_w * S_wc) / (1 - S_wc)
                                    E_fw_drive = B_gi * C_val * (res_p - pressure_data)
                                    We_term_drive = We_data / (
                                            G_intercept * 1e3) if influx_present and G_intercept != 0 else np.zeros(
                                        n_points)
                                    total_expansion_drive = E_g_drive + E_fw_drive + We_term_drive
                                    gas_exp_pct = np.nan_to_num((E_g_drive / total_expansion_drive) * 100)
                                    rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion_drive) * 100)
                                    water_influx_pct = np.nan_to_num((We_term_drive / total_expansion_drive) * 100)
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.stackplot(pressure_data[1:], gas_exp_pct[1:], rock_water_pct[1:],
                                                 water_influx_pct[1:],
                                                 labels=['Gas Expansion', 'Rock & Fluid', 'Water Influx'],
                                                 colors=['red', 'blue', 'green'])
                                    ax.set_xlabel('Pressure (psi)');
                                    ax.set_ylabel('Contribution (%)')
                                    ax.set_title('Drive Mechanism Analysis');
                                    ax.legend();
                                    ax.grid()
                                    ax.invert_xaxis()
                                    st.pyplot(fig)

                            # Step 8: Future Performance Prediction
                            if st.checkbox("Predict Future Performance"):
                                future_date_str = st.text_input("Enter Future Date (YYYY-MM-DD):")
                                if future_date_str:
                                    try:
                                        future_date = pd.to_datetime(future_date_str)
                                        if future_date <= dates.iloc[-1]:
                                            st.error("Future date must be after the last historical date.")
                                        else:
                                            time_days = (dates - dates.iloc[0]).dt.days
                                            time_diffs = np.diff(time_days)
                                            future_dates = pd.date_range(start=dates.iloc[-1], end=future_date,
                                                                         freq='MS')
                                            if future_dates[0] != dates.iloc[-1]:
                                                future_dates = np.insert(future_dates, 0, dates.iloc[-1])
                                            time_to_future = (future_dates - dates.iloc[-1]).days

                                            # Calculate and Display Average Flow Rates
                                            if reservoir_type == "Oil":
                                                q_o = np.diff(Np_data * 1e6) / time_diffs
                                                q_g = np.diff(Gp_data * 1e6) / time_diffs
                                                q_w = np.diff(Wp_data * 1e6) / time_diffs
                                                q_o_avg, q_g_avg, q_w_avg = np.mean(q_o), np.mean(q_g), np.mean(q_w)
                                                st.write(f"Average Oil Production Rate: {q_o_avg:.2f} STB/day")
                                                st.write(f"Average Gas Production Rate: {q_g_avg:.2f} SCF/day")
                                                st.write(f"Average Water Production Rate: {q_w_avg:.2f} STB/day")
                                            else:
                                                q_g = np.diff(Gp_data * 1e9) / time_diffs
                                                q_w = np.diff(Wp_data * 1e6) / time_diffs
                                                q_g_avg, q_w_avg = np.mean(q_g), np.mean(q_w)
                                                st.write(f"Average Gas Production Rate: {q_g_avg:.2f} SCF/day")
                                                st.write(f"Average Water Production Rate: {q_w_avg:.2f} STB/day")

                                            # Future Performance Calculations
                                            if reservoir_type == "Oil":
                                                N_STB = N_intercept * 1e6
                                                Np_future, Gp_future, Wp_future, valid_dates = [], [], [], []
                                                for dt, f_date in zip(time_to_future, future_dates):
                                                    Np_temp = Np_data[-1] * 1e6 + q_o_avg * dt
                                                    if Np_temp > N_STB:
                                                        break
                                                    Gp_temp = Gp_data[-1] * 1e6 + q_g_avg * dt
                                                    Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                                    Np_future.append(Np_temp / 1e6)
                                                    Gp_future.append(Gp_temp / 1e6)
                                                    Wp_future.append(Wp_temp / 1e6)
                                                    valid_dates.append(f_date)
                                                future_dates = valid_dates

                                                def mb_oil(p, Np, Gp, Wp):
                                                    Bo = calculate_Bo(p, p_b, r_s_input, sg_g, api, t)
                                                    Bg = calculate_Bg(p, sg_g, t)
                                                    Bw = calculate_Bw(p, t)
                                                    F = Np * 1e6 * Bo + (
                                                            Gp * 1e6 - Np * 1e6 * R_si) * Bg + Wp * 1e6 * Bw
                                                    E_o = (Bo - B_oi) + (
                                                            R_si - calculate_Rs(p, p_b, r_s_input, sg_g, api,
                                                                                t)) * Bg if p < p_b else Bo - B_oi
                                                    E_g = B_oi * (Bg / B_gi - 1)
                                                    E_fw = (1 + m) * B_oi * c_e * (res_p - p)
                                                    E_t = E_o + m * E_g + E_fw
                                                    W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                                    return F - N_STB * E_t - W_e

                                                pressure_future = []
                                                for Np_val, Gp_val, Wp_val in zip(Np_future, Gp_future, Wp_future):
                                                    try:
                                                        p = brentq(lambda p: mb_oil(p, Np_val, Gp_val, Wp_val), 1000,
                                                                   res_p)
                                                        pressure_future.append(p)
                                                    except:
                                                        pressure_future.append(
                                                            pressure_future[-1] if pressure_future else pressure_data[
                                                                -1])
                                                pressure_future = np.array(pressure_future)
                                                Bo_future = np.array(
                                                    [calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in
                                                     pressure_future])
                                                Rs_future = np.array(
                                                    [calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in
                                                     pressure_future])
                                                Bg_future = np.array(
                                                    [calculate_Bg(p, sg_g, t) for p in pressure_future])
                                                Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                                Viscosity_future = np.array(
                                                    [calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in
                                                     pressure_future])
                                                Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                                            else:
                                                G_SCF = G_intercept * 1e9
                                                Gp_future, Wp_future, valid_dates = [], [], []
                                                for dt, f_date in zip(time_to_future, future_dates):
                                                    Gp_temp = Gp_data[-1] * 1e9 + q_g_avg * dt
                                                    if Gp_temp > G_SCF:
                                                        break
                                                    Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                                    Gp_future.append(Gp_temp / 1e9)
                                                    Wp_future.append(Wp_temp / 1e6)
                                                    valid_dates.append(f_date)
                                                future_dates = valid_dates

                                                def mb_gas(p, Gp, Wp):
                                                    Z = compute_z(sg_g, p, t)
                                                    Bg = 0.02827 * Z * (t + 460) / p
                                                    Bw = calculate_Bw(p, t)
                                                    F = (Gp * 1e9) * Bg + (Wp * 1e6) * Bw
                                                    E_g = Bg - B_gi
                                                    C = (c_f + c_w * S_wc) / (1 - S_wc)
                                                    E_fw = B_gi * C * (res_p - p)
                                                    E_t = E_g + E_fw
                                                    W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                                    return F - G_SCF * E_t - W_e

                                                pressure_future = []
                                                for Gp_val, Wp_val in zip(Gp_future, Wp_future):
                                                    try:
                                                        p = brentq(lambda p: mb_gas(p, Gp_val, Wp_val), 100, res_p)
                                                        pressure_future.append(p)
                                                    except:
                                                        pressure_future.append(
                                                            pressure_future[-1] if pressure_future else pressure_data[
                                                                -1])
                                                pressure_future = np.array(pressure_future)
                                                Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                                                Bg_future = 0.02827 * Z_future * (t + 460) / pressure_future
                                                Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                                Viscosity_future = np.array(
                                                    [calculate_gas_viscosity(p, t, sg_g, z) for p, z in
                                                     zip(pressure_future, Z_future)])

                                            # Combine Historical and Future Data
                                            all_dates = np.append(dates, future_dates)
                                            all_pressure = np.append(pressure_data, pressure_future)
                                            all_Gp = np.append(Gp_data,
                                                               Gp_future if reservoir_type == "Gas" else Gp_future)
                                            all_Wp = np.append(Wp_data, Wp_future)
                                            all_Bg = np.append(Bg_data, Bg_future)
                                            all_Bw = np.append(Bw_data, Bw_future)
                                            all_Viscosity = np.append(Viscosity_data, Viscosity_future)
                                            all_Z = np.append(Z_prod, Z_future)
                                            if reservoir_type == "Oil":
                                                all_Np = np.append(Np_data, Np_future)
                                                all_Bo = np.append(Bo_data, Bo_future)
                                                all_Rs = np.append(Rs_data, Rs_future)

                                            # Plot P/Z vs Gp - Gi for Gas Reservoir
                                            if reservoir_type == "Gas" and st.checkbox(
                                                    "Plot P/Z vs Gp - Gi (Future Prediction)"):
                                                G_initial = Gp_data[0]
                                                all_Gp_minus_Gi = all_Gp - G_initial
                                                all_P_over_Z = all_pressure / all_Z
                                                plt.figure(figsize=(8, 6))
                                                plt.plot(all_Gp_minus_Gi[:len(dates)], all_P_over_Z[:len(dates)], 'bo-',
                                                         linewidth=2,
                                                         label='Historical')
                                                plt.plot(all_Gp_minus_Gi[len(dates):], all_P_over_Z[len(dates):], 'r--',
                                                         linewidth=2,
                                                         label='Future')
                                                plt.xlabel('Gp - Gi (BSCF)', fontweight='bold')
                                                plt.ylabel('P/Z (psi)', fontweight='bold')
                                                plt.title('P/Z vs Gp - Gi (Future Prediction)', fontweight='bold')
                                                plt.legend(prop={'weight': 'bold'})
                                                plt.grid()
                                                st.pyplot(plt)

                                            # Step 9: Plot Reservoir Properties Prediction
                                            if st.checkbox("Plot Reservoir Properties Prediction"):
                                                fig, axs = plt.subplots(3 if reservoir_type == "Oil" else 2, 2,
                                                                        figsize=(
                                                                            14, 15 if reservoir_type == "Oil" else 10))
                                                axs = axs.flatten()
                                                plot_data = (
                                                    [("Bo (RB/STB)", all_Bo, all_Bo[len(dates):]),
                                                     ("Rs (SCF/STB)", all_Rs, all_Rs[len(dates):]),
                                                     ("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]),
                                                     ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                                     ("Viscosity (cP)", all_Viscosity, all_Viscosity[len(dates):]),
                                                     ("Z-factor", all_Z, all_Z[len(dates):])]
                                                    if reservoir_type == "Oil"
                                                    else [("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]),
                                                          ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                                          ("Viscosity (cp)", all_Viscosity, all_Viscosity[len(dates):]),
                                                          ("Z-factor", all_Z, all_Z[len(dates):])]
                                                )
                                                for ax, (label, hist_data, fut_data) in zip(axs, plot_data):
                                                    ax.plot(all_pressure[:len(dates)], hist_data[:len(dates)], 'b-o',
                                                            label=f'Hist {label}')
                                                    ax.plot(all_pressure[len(dates):], fut_data, 'r--',
                                                            label=f'Fut {label}')
                                                    if reservoir_type == "Oil":
                                                        ax.axvline(p_b, color='k', linestyle='--',
                                                                   label=f'Pb: {p_b:.2f} psi')
                                                    ax.set_xlabel('Pressure (psi)');
                                                    ax.set_ylabel(label)
                                                    ax.set_title(f'{label} vs Pressure');
                                                    ax.legend();
                                                    ax.grid()
                                                if reservoir_type == "Oil" and len(axs) > len(plot_data):
                                                    fig.delaxes(axs[-1])
                                                plt.tight_layout()
                                                st.pyplot(fig)

                                            # Step 10: Plot Cumulative Production
                                            if st.checkbox("Plot Cumulative Production"):
                                                if reservoir_type == "Oil":
                                                    # vs Time
                                                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                                    ax1.plot(dates, Np_data, 'b-', label='Cum Oil (Hist)')
                                                    ax1.plot(future_dates, Np_future, 'r--', label='Cum Oil (Fut)')
                                                    ax1.set_xlabel('Date');
                                                    ax1.set_ylabel('Cum Oil (MMSTB)')
                                                    ax1.set_title('Cum Oil vs Time');
                                                    ax1.legend();
                                                    ax1.grid()

                                                    ax2.plot(dates, Gp_data, 'g-', label='Cum Gas (Hist)')
                                                    ax2.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                                    ax2.set_xlabel('Date');
                                                    ax2.set_ylabel('Cum Gas (MMSCF)')
                                                    ax2.set_title('Cum Gas vs Time');
                                                    ax2.legend();
                                                    ax2.grid()

                                                    ax3.plot(dates, Wp_data, 'm-', label='Cum Water (Hist)')
                                                    ax3.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                                    ax3.set_xlabel('Date');
                                                    ax3.set_ylabel('Cum Water (MMSTB)')
                                                    ax3.set_title('Cum Water vs Time');
                                                    ax3.legend();
                                                    ax3.grid()
                                                    plt.tight_layout()
                                                    st.pyplot(fig)

                                                    # vs Pressure
                                                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                                    ax1.plot(pressure_data, Np_data, 'b-', label='Cum Oil (Hist)')
                                                    ax1.plot(pressure_future, Np_future, 'r--', label='Cum Oil (Fut)')
                                                    ax1.set_xlabel('Pressure (psi)');
                                                    ax1.set_ylabel('Cum Oil (MMSTB)')
                                                    ax1.set_title('Cum Oil vs Pressure');
                                                    ax1.legend();
                                                    ax1.grid()

                                                    ax2.plot(pressure_data, Gp_data, 'g-', label='Cum Gas (Hist)')
                                                    ax2.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                                    ax2.set_xlabel('Pressure (psi)');
                                                    ax2.set_ylabel('Cum Gas (MMSCF)')
                                                    ax2.set_title('Cum Gas vs Pressure');
                                                    ax2.legend();
                                                    ax2.grid()

                                                    ax3.plot(pressure_data, Wp_data, 'm-', label='Cum Water (Hist)')
                                                    ax3.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                                    ax3.set_xlabel('Pressure (psi)');
                                                    ax3.set_ylabel('Cum Water (MMSTB)')
                                                    ax3.set_title('Cum Water vs Pressure');
                                                    ax3.legend();
                                                    ax3.grid()
                                                    plt.tight_layout()
                                                    st.pyplot(fig)
                                                else:
                                                    # vs Time
                                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                                    ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (Hist)')
                                                    ax1.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                                    ax1.set_xlabel('Date');
                                                    ax1.set_ylabel('Cum Gas (BSCF)')
                                                    ax1.set_title('Cum Gas vs Time');
                                                    ax1.legend();
                                                    ax1.grid()

                                                    ax2.plot(dates, Wp_data, 'g-', label='Cum Water (Hist)')
                                                    ax2.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                                    ax2.set_xlabel('Date');
                                                    ax2.set_ylabel('Cum Water (MMSTB)')
                                                    ax2.set_title('Cum Water vs Time');
                                                    ax2.legend();
                                                    ax2.grid()
                                                    plt.tight_layout()
                                                    st.pyplot(fig)

                                                    # vs Pressure
                                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                                    ax1.plot(pressure_data, Gp_data, 'b-', label='Cum Gas (Hist)')
                                                    ax1.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                                    ax1.set_xlabel('Pressure (psi)');
                                                    ax1.set_ylabel('Cum Gas (BSCF)')
                                                    ax1.set_title('Cum Gas vs Pressure');
                                                    ax1.legend();
                                                    ax1.grid()

                                                    ax2.plot(pressure_data, Wp_data, 'g-', label='Cum Water (Hist)')
                                                    ax2.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                                    ax2.set_xlabel('Pressure (psi)');
                                                    ax2.set_ylabel('Cum Water (MMSTB)')
                                                    ax2.set_title('Cum Water vs Pressure');
                                                    ax2.legend();
                                                    ax2.grid()
                                                    plt.tight_layout()
                                                    st.pyplot(fig)

                                            # Save Data
                                            final_output = pd.DataFrame({
                                                'Date': all_dates,
                                                'Pressure (psi)': all_pressure,
                                                'Cum Gas Production': all_Gp,
                                                'Cum Water Production (MMSTB)': all_Wp,
                                                'Bg (RB/SCF)': all_Bg,
                                                'Bw (RB/STB)': all_Bw,
                                                'Viscosity': all_Viscosity,
                                                'Z-factor': all_Z
                                            })
                                            if reservoir_type == "Oil":
                                                final_output['Cum Oil Production (MMSTB)'] = all_Np
                                                final_output['Bo (RB/STB)'] = all_Bo
                                                final_output['Rs (SCF/STB)'] = all_Rs
                                                final_output.rename(
                                                    columns={'Cum Gas Production': 'Cum Gas Production (MMSCF)',
                                                             'Viscosity': 'Viscosity (cP)'}, inplace=True)
                                            else:
                                                final_output.rename(
                                                    columns={'Cum Gas Production': 'Cum Gas Production (BSCF)',
                                                             'Viscosity': 'Viscosity (cp)'}, inplace=True)
                                            csv = final_output.to_csv(index=False)
                                            st.download_button("Download Results", csv,
                                                               f"{reservoir_type.lower()}_material_balance.csv",
                                                               "text/csv")
                                    except ValueError:
                                        st.error("Invalid date format. Use YYYY-MM-DD.")
                            else:
                                if data_source == "Upload File":
                                    st.write("Please upload a production data file.")
                                elif data_source == "Use Example Data":
                                    st.write("Example data file not found. Please upload a production data file.")
    else:
        # Data source selection and file handling
        data_source = st.selectbox("1.Select Data Source", ["Upload File", "Use Example Data"], key="data_source")
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("2.Upload Production Data (Excel)", type=["xls", "xlsx"], )
            st.write(
                ">>*NOTE*>> The excel file should have column names: Date, Pressure, Cum Gas Production (BSCF), and optionally Cum Water Production (MMSTB).")
            if uploaded_file:
                history_data = pd.read_excel(uploaded_file)
            else:
                history_data = None
        else:
            try:
                history_data = pd.read_excel("example_gas_data.xlsx")
                st.write(f"Using example data for {reservoir_type} reservoir.")
            except FileNotFoundError:
                st.error(f"Example data file for {reservoir_type} not found.")
                history_data = None

        if history_data is not None:
            required_columns = (
                ['Date', 'Pressure', 'Cum Oil Production', 'Cum Gas Production', 'Cum Water Production']
                if reservoir_type == "Oil"
                else ['Date', 'Pressure', 'Cum Gas Production']
            )
            missing_columns = [col for col in required_columns if col not in history_data.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                # Data inputs

                st.sidebar.header("Reservoir Parameters")
                res_p = st.sidebar.number_input("Reservoir Pressure (psi)", min_value=0.0, value=2500.0, step=50.0)
                t = st.sidebar.number_input("Temperature (F)", min_value=0.0, value=200.0, step=5.0)
                sg_g = st.sidebar.number_input("Gas Specific Gravity", min_value=0.0, value=0.75, step=0.05)
                c_f = st.sidebar.number_input("Rock Compressibility (1/psi)", min_value=0.0, value=6e-6, format="%.2e",
                                              step=0.5e-6)
                c_w = st.sidebar.number_input("Water Compressibility (1/psi)", min_value=0.0, value=1e-6, format="%.2e",
                                              step=0.5e-6)
                S_wc = st.sidebar.number_input("Initial Water Saturation (fraction)", min_value=0.0, max_value=1.0,
                                               value=0.15, step=0.05)
                api = st.sidebar.number_input("API Gravity", min_value=0.0, value=20.0, step=5.0)
                r_s_input = st.sidebar.number_input("Solution Gas-Oil Ratio (SCF/STB)", min_value=0.0, value=200.0,
                                                    step=50.0)
                #m = st.sidebar.number_input("Gas Cap Ratio (m)", min_value=0.0, value=0.1, step=0.1)
                dates = pd.to_datetime(history_data['Date'])
                pressure_data = history_data['Pressure'].values
                Gp_data = history_data['Cum Gas Production'].values  # BSCF
                Wp_data = history_data.get('Cum Water Production', np.zeros(len(pressure_data))).values  # MMSTB
                #Np_data = history_data['Cum Oil Production'].values  # MMSTB
                #Gp_data = history_data['Cum Gas Production'].values  # MMSCF
                #Wp_data = history_data['Cum Water Production'].values  # MMSTB
                n_points = len(pressure_data)
                # Fluid Property Calculations
                p_b = 18.2 * (((r_s_input / sg_g) ** 0.83) * (10 ** (0.00091 * t - 0.0125 * api)) - 1.4)
                st.write(f"Calculated Bubble Point Pressure: {p_b:.2f} psi")
                Rs_data = np.array([calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Bo_data = np.array([calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                Bg_data = np.array([calculate_Bg(p, sg_g, t) for p in pressure_data])
                Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                Viscosity_data = np.array(
                    [calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in pressure_data])
                B_oi, R_si, B_gi = Bo_data[0], Rs_data[0], Bg_data[0]
                # else:
                Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                Bg_data = 0.02827 * Z_prod * (t + 460) / pressure_data
                Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                Viscosity_data = np.array(
                    [calculate_gas_viscosity(p, t, sg_g, z) for p, z in zip(pressure_data, Z_prod)])
                B_gi = Bg_data[0]

                # Step 3: Option to Plot Reservoir Plots
                if st.checkbox("Plot Reservoir Production Data"):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (BSCF)')
                    ax1.set_xlabel('Time');
                    ax1.set_ylabel('Cum Gas (BSCF)')
                    ax1.legend();
                    ax1.grid()
                    ax1_twin = ax1.twiny()
                    ax1_twin.plot(pressure_data, Gp_data, 'r--', alpha=0)
                    ax1_twin.set_xlabel('Pressure (psi)');
                    ax1_twin.invert_xaxis()

                    ax2.plot(dates, Wp_data, 'g-', label='Cum Water (MMSTB)')
                    ax2.set_xlabel('Time');
                    ax2.set_ylabel('Cum Water (MMSTB)')
                    ax2.legend();
                    ax2.grid()
                    ax2_twin = ax2.twiny()
                    ax2_twin.plot(pressure_data, Wp_data, 'r--', alpha=0)
                    ax2_twin.set_xlabel('Pressure (psi)');
                    ax2_twin.invert_xaxis()
                    plt.tight_layout()
                    st.pyplot(fig)

                if st.checkbox("Plot Fluid Properties"):
                    plots = (
                        [("Rs (SCF/STB)", Rs_data), ("Bo (RB/STB)", Bo_data), ("Bg (RB/SCF)", Bg_data),
                         ("Bw (RB/STB)", Bw_data), ("Viscosity (cP)", Viscosity_data), ("Z-factor", Z_prod)]
                        if reservoir_type == "Oil"
                        else [("Bg (RB/SCF)", Bg_data), ("Bw (RB/STB)", Bw_data), ("Viscosity (cp)", Viscosity_data),
                              ("Z-factor", Z_prod)]
                    )
                    fig, axs = plt.subplots((len(plots) + 1) // 2, 2, figsize=(12, 6 * ((len(plots) + 1) // 2)))
                    axs = axs.flatten()
                    for ax, (label, data) in zip(axs, plots):
                        ax.plot(pressure_data, data, 'b-', label=label)
                        if reservoir_type == "Oil":
                            ax.axvline(p_b, color='r', linestyle='--', label=f'Pb: {p_b:.2f} psi')
                        ax.set_xlabel('Pressure (psi)');
                        ax.set_ylabel(label)
                        ax.legend();
                        ax.grid()
                    if len(plots) % 2:
                        fig.delaxes(axs[-1])
                    plt.tight_layout()
                    st.pyplot(fig)

                # Step 4: Campbell/Cole Plots
                if st.checkbox("Plot Cole Plot"):
                    cole_y = np.full(n_points, np.nan)
                    for i in range(1, n_points):
                        if Bg_data[i] != B_gi:
                            cole_y[i] = Gp_data[i] * Bg_data[i] / (Bg_data[i] - B_gi)
                    fig, ax = plt.subplots()
                    ax.plot(Gp_data[1:], cole_y[1:], 'bo-', label='Cole Plot')
                    ax.set_xlabel('Cumulative Gas (BSCF)');
                    ax.set_ylabel(r'$\frac{G_p\,Bg}{Bg - B_{gi}}$')
                    ax.set_title('Cole Plot');
                    ax.legend();
                    ax.grid()
                    st.pyplot(fig)

                # Step 5: Water Influx
                influx_present = st.selectbox("Is water influx present?", ["No", "Yes"]) == "Yes"
                if influx_present:
                    W_ei = st.number_input("Aquifer Volume (MMft^3)", min_value=0.0, value=0.0)
                    c_t = c_w + c_f
                    We_data = (W_ei / 5.615) * c_t * (res_p - pressure_data)
                    if st.checkbox("Plot Water Influx"):
                        fig, ax = plt.subplots()
                        ax.plot(dates, We_data, 'r--', label='Water Influx (MMBBL)')
                        ax.set_xlabel('Time');
                        ax.set_ylabel('Water Influx (MMBBL)')
                        ax.set_title('Water Influx Over Time');
                        ax.legend();
                        ax.grid()
                        st.pyplot(fig)
                else:
                    We_data = np.zeros(n_points)

                # Step 6: Material Balance Calculations
                if st.checkbox("Calculate Material Balance"):
                    F_data = (Gp_data * 1e9) * Bg_data + (Wp_data * 1e6) * Bw_data
                    C = (c_f + c_w * S_wc) / (1 - S_wc)
                    E_t_data = (Bg_data - B_gi) + B_gi * C * (res_p - pressure_data)
                    G_intercept_SCF = np.sum(E_t_data * F_data) / np.sum(E_t_data ** 2)
                    G_intercept = G_intercept_SCF / 1e9
                    if not influx_present:
                        ratio = F_data / E_t_data
                        fig, ax = plt.subplots()
                        ax.plot(Gp_data, ratio, 'bo-', label=f'OGIP = {G_intercept:.2f} BSCF')
                        ax.set_xlabel('Cum Gas (BSCF)');
                        ax.set_ylabel('F / E_t (RB)')
                        ax.set_title('Material Balance (No Influx)');
                        ax.legend();
                        ax.grid()
                        st.pyplot(fig)
                        st.write(f"OGIP (No Influx): {G_intercept:.2f} BSCF")
                    else:
                        ratio = (F_data - We_data) / E_t_data
                        fig, ax = plt.subplots()
                        ax.plot(Gp_data, ratio, 'bo-', label=f'OGIP = {G_intercept:.2f} BSCF')
                        ax.set_xlabel('Cum Gas (BSCF)');
                        ax.set_ylabel('(F - W_e) / E_t (RB)')
                        ax.set_title('Material Balance (With Influx)');
                        ax.legend();
                        ax.grid()
                        st.pyplot(fig)
                        st.write(f"OGIP (With Influx): {G_intercept:.2f} BSCF")
                    G_initial = Gp_data[0]
                    model = LinearRegression().fit((Gp_data - G_initial).reshape(-1, 1), pressure_data / Z_prod)
                    OGIP_estimated = model.intercept_ / -model.coef_[0]
                    fig, ax = plt.subplots()
                    ax.plot(Gp_data - G_initial, pressure_data / Z_prod, 'bo', label='Data')
                    x_fit = np.linspace(min(Gp_data - G_initial), max(Gp_data - G_initial), 100)
                    ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), 'r-', label=f'OGIP = {OGIP_estimated:.2f} BSCF')
                    ax.set_xlabel('Gp - Gi (BSCF)');
                    ax.set_ylabel('p/Z (psi)')
                    ax.set_title('Modified p/Z Plot');
                    ax.legend();
                    ax.grid()
                    st.pyplot(fig)

                     # Step 7: Drive Mechanism Analysis
                    if st.checkbox("Plot Drive Mechanism Analysis"):
                        E_g_drive = Bg_data - B_gi
                        C_val = (c_f + c_w * S_wc) / (1 - S_wc)
                        E_fw_drive = B_gi * C_val * (res_p - pressure_data)
                        We_term_drive = We_data / (
                                G_intercept * 1e3) if influx_present and G_intercept != 0 else np.zeros(
                            n_points)
                        total_expansion_drive = E_g_drive + E_fw_drive + We_term_drive
                        gas_exp_pct = np.nan_to_num((E_g_drive / total_expansion_drive) * 100)
                        rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion_drive) * 100)
                        water_influx_pct = np.nan_to_num((We_term_drive / total_expansion_drive) * 100)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.stackplot(pressure_data[1:], gas_exp_pct[1:], rock_water_pct[1:],
                                     water_influx_pct[1:],
                                     labels=['Gas Expansion', 'Rock & Fluid', 'Water Influx'],
                                     colors=['red', 'blue', 'green'])
                        ax.set_xlabel('Pressure (psi)');
                        ax.set_ylabel('Contribution (%)')
                        ax.set_title('Drive Mechanism Analysis');
                        ax.legend();
                        ax.grid()
                        ax.invert_xaxis()
                        st.pyplot(fig)

                # Step 8: Future Performance Prediction
                if st.checkbox("Predict Future Performance"):
                    future_date_str = st.text_input("Enter Future Date (YYYY-MM-DD):")
                    if future_date_str:
                        try:
                            future_date = pd.to_datetime(future_date_str)
                            if future_date <= dates.iloc[-1]:
                                st.error("Future date must be after the last historical date.")
                            else:
                                time_days = (dates - dates.iloc[0]).dt.days
                                time_diffs = np.diff(time_days)
                                future_dates = pd.date_range(start=dates.iloc[-1], end=future_date,
                                                             freq='MS')
                                if future_dates[0] != dates.iloc[-1]:
                                    future_dates = np.insert(future_dates, 0, dates.iloc[-1])
                                time_to_future = (future_dates - dates.iloc[-1]).days

                                # Calculate and Display Average Flow Rates
                                if reservoir_type == "Oil":
                                    q_o = np.diff(Np_data * 1e6) / time_diffs
                                    q_g = np.diff(Gp_data * 1e6) / time_diffs
                                    q_w = np.diff(Wp_data * 1e6) / time_diffs
                                    q_o_avg, q_g_avg, q_w_avg = np.mean(q_o), np.mean(q_g), np.mean(q_w)
                                    st.write(f"Average Oil Production Rate: {q_o_avg:.2f} STB/day")
                                    st.write(f"Average Gas Production Rate: {q_g_avg:.2f} SCF/day")
                                    st.write(f"Average Water Production Rate: {q_w_avg:.2f} STB/day")
                                else:
                                    q_g = np.diff(Gp_data * 1e9) / time_diffs
                                    q_w = np.diff(Wp_data * 1e6) / time_diffs
                                    q_g_avg, q_w_avg = np.mean(q_g), np.mean(q_w)
                                    st.write(f"Average Gas Production Rate: {q_g_avg:.2f} SCF/day")
                                    st.write(f"Average Water Production Rate: {q_w_avg:.2f} STB/day")

                                # Future Performance Calculations
                                if reservoir_type == "Oil":
                                    N_STB = N_intercept * 1e6
                                    Np_future, Gp_future, Wp_future, valid_dates = [], [], [], []
                                    for dt, f_date in zip(time_to_future, future_dates):
                                        Np_temp = Np_data[-1] * 1e6 + q_o_avg * dt
                                        if Np_temp > N_STB:
                                            break
                                        Gp_temp = Gp_data[-1] * 1e6 + q_g_avg * dt
                                        Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                        Np_future.append(Np_temp / 1e6)
                                        Gp_future.append(Gp_temp / 1e6)
                                        Wp_future.append(Wp_temp / 1e6)
                                        valid_dates.append(f_date)
                                    future_dates = valid_dates

                                    def mb_oil(p, Np, Gp, Wp):
                                        Bo = calculate_Bo(p, p_b, r_s_input, sg_g, api, t)
                                        Bg = calculate_Bg(p, sg_g, t)
                                        Bw = calculate_Bw(p, t)
                                        F = Np * 1e6 * Bo + (
                                                Gp * 1e6 - Np * 1e6 * R_si) * Bg + Wp * 1e6 * Bw
                                        E_o = (Bo - B_oi) + (
                                                R_si - calculate_Rs(p, p_b, r_s_input, sg_g, api,
                                                                    t)) * Bg if p < p_b else Bo - B_oi
                                        E_g = B_oi * (Bg / B_gi - 1)
                                        E_fw = (1 + m) * B_oi * c_e * (res_p - p)
                                        E_t = E_o + m * E_g + E_fw
                                        W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                        return F - N_STB * E_t - W_e

                                    pressure_future = []
                                    for Np_val, Gp_val, Wp_val in zip(Np_future, Gp_future, Wp_future):
                                        try:
                                            p = brentq(lambda p: mb_oil(p, Np_val, Gp_val, Wp_val), 1000,
                                                       res_p)
                                            pressure_future.append(p)
                                        except:
                                            pressure_future.append(
                                                pressure_future[-1] if pressure_future else pressure_data[
                                                    -1])
                                    pressure_future = np.array(pressure_future)
                                    Bo_future = np.array(
                                        [calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in
                                         pressure_future])
                                    Rs_future = np.array(
                                        [calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in
                                         pressure_future])
                                    Bg_future = np.array(
                                        [calculate_Bg(p, sg_g, t) for p in pressure_future])
                                    Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                    Viscosity_future = np.array(
                                        [calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in
                                         pressure_future])
                                    Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                                else:
                                    G_SCF = G_intercept * 1e9
                                    Gp_future, Wp_future, valid_dates = [], [], []
                                    for dt, f_date in zip(time_to_future, future_dates):
                                        Gp_temp = Gp_data[-1] * 1e9 + q_g_avg * dt
                                        if Gp_temp > G_SCF:
                                            break
                                        Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                        Gp_future.append(Gp_temp / 1e9)
                                        Wp_future.append(Wp_temp / 1e6)
                                        valid_dates.append(f_date)
                                    future_dates = valid_dates

                                    def mb_gas(p, Gp, Wp):
                                        Z = compute_z(sg_g, p, t)
                                        Bg = 0.02827 * Z * (t + 460) / p
                                        Bw = calculate_Bw(p, t)
                                        F = (Gp * 1e9) * Bg + (Wp * 1e6) * Bw
                                        E_g = Bg - B_gi
                                        C = (c_f + c_w * S_wc) / (1 - S_wc)
                                        E_fw = B_gi * C * (res_p - p)
                                        E_t = E_g + E_fw
                                        W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                        return F - G_SCF * E_t - W_e

                                    pressure_future = []
                                    for Gp_val, Wp_val in zip(Gp_future, Wp_future):
                                        try:
                                            p = brentq(lambda p: mb_gas(p, Gp_val, Wp_val), 100, res_p)
                                            pressure_future.append(p)
                                        except:
                                            pressure_future.append(
                                                pressure_future[-1] if pressure_future else pressure_data[
                                                    -1])
                                    pressure_future = np.array(pressure_future)
                                    Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                                    Bg_future = 0.02827 * Z_future * (t + 460) / pressure_future
                                    Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                    Viscosity_future = np.array(
                                        [calculate_gas_viscosity(p, t, sg_g, z) for p, z in
                                         zip(pressure_future, Z_future)])

                                # Combine Historical and Future Data
                                all_dates = np.append(dates, future_dates)
                                all_pressure = np.append(pressure_data, pressure_future)
                                all_Gp = np.append(Gp_data,
                                                   Gp_future if reservoir_type == "Gas" else Gp_future)
                                all_Wp = np.append(Wp_data, Wp_future)
                                all_Bg = np.append(Bg_data, Bg_future)
                                all_Bw = np.append(Bw_data, Bw_future)
                                all_Viscosity = np.append(Viscosity_data, Viscosity_future)
                                all_Z = np.append(Z_prod, Z_future)
                                if reservoir_type == "Oil":
                                    all_Np = np.append(Np_data, Np_future)
                                    all_Bo = np.append(Bo_data, Bo_future)
                                    all_Rs = np.append(Rs_data, Rs_future)

                                # Plot P/Z vs Gp - Gi for Gas Reservoir
                                if reservoir_type == "Gas" and st.checkbox(
                                        "Plot P/Z vs Gp - Gi (Future Prediction)"):
                                    G_initial = Gp_data[0]
                                    all_Gp_minus_Gi = all_Gp - G_initial
                                    all_P_over_Z = all_pressure / all_Z
                                    plt.figure(figsize=(8, 6))
                                    plt.plot(all_Gp_minus_Gi[:len(dates)], all_P_over_Z[:len(dates)], 'bo-',
                                             linewidth=2,
                                             label='Historical')
                                    plt.plot(all_Gp_minus_Gi[len(dates):], all_P_over_Z[len(dates):], 'r--',
                                             linewidth=2,
                                             label='Future')
                                    plt.xlabel('Gp - Gi (BSCF)', fontweight='bold')
                                    plt.ylabel('P/Z (psi)', fontweight='bold')
                                    plt.title('P/Z vs Gp - Gi (Future Prediction)', fontweight='bold')
                                    plt.legend(prop={'weight': 'bold'})
                                    plt.grid()
                                    st.pyplot(plt)

                                # Step 9: Plot Reservoir Properties Prediction
                                if st.checkbox("Plot Reservoir Properties Prediction"):
                                    fig, axs = plt.subplots(3 if reservoir_type == "Oil" else 2, 2,
                                                            figsize=(
                                                                14, 15 if reservoir_type == "Oil" else 10))
                                    axs = axs.flatten()
                                    plot_data = (
                                        [("Bo (RB/STB)", all_Bo, all_Bo[len(dates):]),
                                         ("Rs (SCF/STB)", all_Rs, all_Rs[len(dates):]),
                                         ("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]),
                                         ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                         ("Viscosity (cP)", all_Viscosity, all_Viscosity[len(dates):]),
                                         ("Z-factor", all_Z, all_Z[len(dates):])]
                                        if reservoir_type == "Oil"
                                        else [("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]),
                                              ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                              ("Viscosity (cp)", all_Viscosity, all_Viscosity[len(dates):]),
                                              ("Z-factor", all_Z, all_Z[len(dates):])]
                                    )
                                    for ax, (label, hist_data, fut_data) in zip(axs, plot_data):
                                        ax.plot(all_pressure[:len(dates)], hist_data[:len(dates)], 'b-o',
                                                label=f'Hist {label}')
                                        ax.plot(all_pressure[len(dates):], fut_data, 'r--',
                                                label=f'Fut {label}')
                                        if reservoir_type == "Oil":
                                            ax.axvline(p_b, color='k', linestyle='--',
                                                       label=f'Pb: {p_b:.2f} psi')
                                        ax.set_xlabel('Pressure (psi)');
                                        ax.set_ylabel(label)
                                        ax.set_title(f'{label} vs Pressure');
                                        ax.legend();
                                        ax.grid()
                                    if reservoir_type == "Oil" and len(axs) > len(plot_data):
                                        fig.delaxes(axs[-1])
                                    plt.tight_layout()
                                    st.pyplot(fig)

                                # Step 10: Plot Cumulative Production
                                if st.checkbox("Plot Cumulative Production"):
                                    if reservoir_type == "Oil":
                                        # vs Time
                                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                        ax1.plot(dates, Np_data, 'b-', label='Cum Oil (Hist)')
                                        ax1.plot(future_dates, Np_future, 'r--', label='Cum Oil (Fut)')
                                        ax1.set_xlabel('Date');
                                        ax1.set_ylabel('Cum Oil (MMSTB)')
                                        ax1.set_title('Cum Oil vs Time');
                                        ax1.legend();
                                        ax1.grid()

                                        ax2.plot(dates, Gp_data, 'g-', label='Cum Gas (Hist)')
                                        ax2.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                        ax2.set_xlabel('Date');
                                        ax2.set_ylabel('Cum Gas (MMSCF)')
                                        ax2.set_title('Cum Gas vs Time');
                                        ax2.legend();
                                        ax2.grid()

                                        ax3.plot(dates, Wp_data, 'm-', label='Cum Water (Hist)')
                                        ax3.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                        ax3.set_xlabel('Date');
                                        ax3.set_ylabel('Cum Water (MMSTB)')
                                        ax3.set_title('Cum Water vs Time');
                                        ax3.legend();
                                        ax3.grid()
                                        plt.tight_layout()
                                        st.pyplot(fig)

                                        # vs Pressure
                                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                        ax1.plot(pressure_data, Np_data, 'b-', label='Cum Oil (Hist)')
                                        ax1.plot(pressure_future, Np_future, 'r--', label='Cum Oil (Fut)')
                                        ax1.set_xlabel('Pressure (psi)');
                                        ax1.set_ylabel('Cum Oil (MMSTB)')
                                        ax1.set_title('Cum Oil vs Pressure');
                                        ax1.legend();
                                        ax1.grid()

                                        ax2.plot(pressure_data, Gp_data, 'g-', label='Cum Gas (Hist)')
                                        ax2.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                        ax2.set_xlabel('Pressure (psi)');
                                        ax2.set_ylabel('Cum Gas (MMSCF)')
                                        ax2.set_title('Cum Gas vs Pressure');
                                        ax2.legend();
                                        ax2.grid()

                                        ax3.plot(pressure_data, Wp_data, 'm-', label='Cum Water (Hist)')
                                        ax3.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                        ax3.set_xlabel('Pressure (psi)');
                                        ax3.set_ylabel('Cum Water (MMSTB)')
                                        ax3.set_title('Cum Water vs Pressure');
                                        ax3.legend();
                                        ax3.grid()
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    else:
                                        # vs Time
                                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                        ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (Hist)')
                                        ax1.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                        ax1.set_xlabel('Date');
                                        ax1.set_ylabel('Cum Gas (BSCF)')
                                        ax1.set_title('Cum Gas vs Time');
                                        ax1.legend();
                                        ax1.grid()

                                        ax2.plot(dates, Wp_data, 'g-', label='Cum Water (Hist)')
                                        ax2.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                        ax2.set_xlabel('Date');
                                        ax2.set_ylabel('Cum Water (MMSTB)')
                                        ax2.set_title('Cum Water vs Time');
                                        ax2.legend();
                                        ax2.grid()
                                        plt.tight_layout()
                                        st.pyplot(fig)

                                        # vs Pressure
                                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                        ax1.plot(pressure_data, Gp_data, 'b-', label='Cum Gas (Hist)')
                                        ax1.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                        ax1.set_xlabel('Pressure (psi)');
                                        ax1.set_ylabel('Cum Gas (BSCF)')
                                        ax1.set_title('Cum Gas vs Pressure');
                                        ax1.legend();
                                        ax1.grid()

                                        ax2.plot(pressure_data, Wp_data, 'g-', label='Cum Water (Hist)')
                                        ax2.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                        ax2.set_xlabel('Pressure (psi)');
                                        ax2.set_ylabel('Cum Water (MMSTB)')
                                        ax2.set_title('Cum Water vs Pressure');
                                        ax2.legend();
                                        ax2.grid()
                                        plt.tight_layout()
                                        st.pyplot(fig)

                                # Save Data
                                final_output = pd.DataFrame({
                                    'Date': all_dates,
                                    'Pressure (psi)': all_pressure,
                                    'Cum Gas Production': all_Gp,
                                    'Cum Water Production (MMSTB)': all_Wp,
                                    'Bg (RB/SCF)': all_Bg,
                                    'Bw (RB/STB)': all_Bw,
                                    'Viscosity': all_Viscosity,
                                    'Z-factor': all_Z
                                })
                                if reservoir_type == "Oil":
                                    final_output['Cum Oil Production (MMSTB)'] = all_Np
                                    final_output['Bo (RB/STB)'] = all_Bo
                                    final_output['Rs (SCF/STB)'] = all_Rs
                                    final_output.rename(
                                        columns={'Cum Gas Production': 'Cum Gas Production (MMSCF)',
                                                 'Viscosity': 'Viscosity (cP)'}, inplace=True)
                                else:
                                    final_output.rename(
                                        columns={'Cum Gas Production': 'Cum Gas Production (BSCF)',
                                                 'Viscosity': 'Viscosity (cp)'}, inplace=True)
                                csv = final_output.to_csv(index=False)
                                st.download_button("Download Results", csv,
                                                   f"{reservoir_type.lower()}_material_balance.csv",
                                                   "text/csv")
                        except ValueError:
                            st.error("Invalid date format. Use YYYY-MM-DD.")
                
        else:
            if data_source == "Upload File":
                st.write("Please upload a production data file.")
            elif data_source == "Use Example Data":
                st.write("Example data file not found. Please upload a production data file.")


if __name__ == "__main__":
    main()
