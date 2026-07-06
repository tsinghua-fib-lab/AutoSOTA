function [E_O, E_O_m_diff] = E_Overlap(rho_ast)

[V, E_edges] = size(rho_ast); % V: number of vertices, E: number of edges

% E_O
E_He_given_z = rho_ast; % V x E (given z=1)
E_He_given_z_norm1 = sum(E_He_given_z, 1); % 1 x E
E_O = zeros(E_edges, E_edges); % E x E
E_O_m_diff = zeros(E_edges, E_edges, V); % E x E x V
for e1 = 1:E_edges
    E_He_given_z_e1 = E_He_given_z(:, e1);
    for e2 = 1:E_edges
        if e2 == e1
            continue; % Skip self-overlap
        end
        % E[O_{e1,e2} | z_e=1, z_e'=1]
        tmp = E_He_given_z(:, e1)' * E_He_given_z(:, e2); % Scalar
        m1 = min(E_He_given_z_norm1(e1), E_He_given_z_norm1(e2));
        if m1 == 0
            E_O(e1, e2) = 0;
        else
            E_O(e1, e2) = tmp / m1; % Scalar
        end

        if nargout > 1
            % E[O_{e1, e2} | z_e=1, m_{v,e}=1] - E[O_{e1, e2} | z_e=1, m_{v,e}=0]
            for v = 1:V
                E_He_given_z_e1_v0 = E_He_given_z_e1; E_He_given_z_e1_v0(v) = 0; % Zero out v-th element
                he_v0_E_Hl = E_He_given_z_e1_v0' * E_He_given_z(:, e2);
                E_He_given_z_e1_v0_norm = sum(E_He_given_z_e1_v0);
                E_He_given_z_e1_v1_norm = E_He_given_z_e1_v0_norm + 1;
                m1_v0 = min(E_He_given_z_e1_v0_norm, E_He_given_z_norm1(e2));
                m1_v1 = min(E_He_given_z_e1_v1_norm, E_He_given_z_norm1(e2));
                if m1_v0 == 0
                    val_v0 = 0;
                else
                    val_v0 = he_v0_E_Hl / m1_v0;
                end
                
                if m1_v1 == 0
                    val_v1 = 0;
                else
                    val_v1 = (he_v0_E_Hl + E_He_given_z(v, e2)) / m1_v1;
                end
                
                E_O_m_diff(e1, e2, v) = val_v1 - val_v0; % Scalar
            end
        end
        
    end
end

end