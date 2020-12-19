// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once
#include "hpg_config.hpp"
#include "hpg_error.hpp"
#include "hpg_export.h"

#include <memory>
#include <tuple>
#if HPG_API >= 17
# include <variant>
#endif

namespace hpg {

/** type for containing a value or an error
 *
 * normally appears as a return value type for a method or function that can
 * fail
 */
#if HPG_API >= 17
template <typename T>
using rval_t = std::variant<Error, T>;
#else // HPG_API < 17
template <typename T>
using rval_t = std::tuple<std::unique_ptr<Error>, T>;
#endif // HPG_API >= 17

/** query whether rval_t value contains an error
 */
template <typename T>
HPG_EXPORT inline bool
is_error(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<Error>(rv);
#else // HPG_API < 17
  return bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** type trait to get type parameter of a template type
 */
template <typename T>
struct HPG_EXPORT rval_value {
  using type = void; /**< parameter type */
};
/** specialized type trait to get value type from rval_t type
 */
template <typename T>
struct rval_value<rval_t<T>> {
  using type = T; /**< parameter type */
};

/** query whether rval_t value contains a (non-error) value
 */
template <typename T>
HPG_EXPORT inline bool
is_value(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<T>(rv);
#else // HPG_API < 17
  return !bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** get value from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_value(RV&& rv) {
  return std::get<1>(std::forward<RV>(rv));
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_value(const RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_value(RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_value(RV&& rv) {
  return std::get<1>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** get error from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_error(RV&& rv) {
# if HPG_API >= 17
  return std::get<0>(std::forward<RV>(rv));
# else
  return *std::get<0>(std::forward<RV>(rv));
# endif
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_error(const RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_error(RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_error(RV&& rv) {
  return *std::get<0>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** create an rval_t value from a (non-error) value
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(T&& t) {
#if HPG_API >= 17
  return rval_t<T>(std::forward<T>(t));
#else // HPG_API < 17
  return {std::unique_ptr<Error>(), std::forward<T>(t)};
#endif // HPG_API >= 17
}

/** create an rval_t value from an Error
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(const Error& err) {
#if HPG_API >= 17
  return rval_t<T>(err);
#else // HPG_API < 17
  return {std::unique_ptr<Error>(new Error(err)), T()};
#endif // HPG_API >= 17
}

/** apply function that returns a plain value to an rval_t
 */
#if HPG_API >= 17 || defined(HPG_INTERNAL)
struct HPG_EXPORT RvalMM {

  template <typename T, typename F>
  static rval_t<std::invoke_result_t<F, T>>
  map(const rval_t<T>& rv, F f) {
    if (is_value(rv))
      return rval<std::invoke_result_t<F, T>>(f(get_value(rv)));
    else
      return rval<std::invoke_result_t<F, T>>(get_error(rv));
  }

  template <typename T, typename F>
  static rval_t<std::invoke_result_t<F, T>>
  map(rval_t<T>&& rv, F f) {
    if (is_value(rv))
      return rval<std::invoke_result_t<F, T>>(f(get_value(std::move(rv))));
    else
      return rval<std::invoke_result_t<F, T>>(get_error(std::move(rv)));
  }

  /** apply function that returns an rval_t value to an rval_t
   */
  template <typename T, typename F>
  static rval_t<typename rval_value<std::invoke_result_t<F, T>>::type>
  flatmap(const rval_t<T>& rv, F f) {
    if (is_value(rv))
      return f(get_value(rv));
    else
      return
        rval<typename rval_value<std::invoke_result_t<F, T>>::type>(
          get_error(rv));
  }

  template <typename T, typename F>
  static rval_t<typename rval_value<std::invoke_result_t<F, T>>::type>
  flatmap(rval_t<T>&& rv, F f) {
    if (is_value(rv))
      return f(get_value(std::move(rv)));
    else
      return
        rval<typename rval_value<std::invoke_result_t<F, T>>::type>(
          get_error(std::move(rv)));
  }

  /** apply function depending on contained value type with common result type
   */
  template <typename T, typename ValF, typename ErrF>
  static std::invoke_result_t<ValF, T>
  fold(const rval_t<T>& rv, ValF vf, ErrF ef) {
    static_assert(
      std::is_same_v<
        std::invoke_result_t<ValF, T>,
        std::invoke_result_t<ErrF, Error>>);

    if (is_value(rv))
      return vf(get_value(rv));
    else
      return ef(get_error(rv));
  }

  template <typename T, typename ValF, typename ErrF>
  static std::invoke_result_t<ValF, T>
  fold(rval_t<T>&& rv, ValF vf, ErrF ef) {
    static_assert(
      std::is_same_v<
        std::invoke_result_t<ValF, T>,
        std::invoke_result_t<ErrF, Error>>);

    if (is_value(rv))
      return vf(get_value(std::move(rv)));
    else
      return ef(get_error(std::move(rv)));
  }
};

template <typename Derived, template <typename> typename M>
struct MonadBase {

  template <typename T>
  static M<T>
  flatten(const M<M<T>>& mmt) {
    return Derived::flat_map(mmt, [](auto&& mt) { return mt; });
  }

  template <typename T>
  static M<T>
  flatten(M<M<T>>&& mmt) {
    return Derived::flat_map(mmt, [](auto&& mt) { return std::move(mt); });
  }

  template <typename F>
  static auto
  lift(F f) {
    return
      [f]<typename MA>(MA&& ma) {
        return map(std::forward<MA>(ma), f);
      };
  }

  template <
    typename T,
    typename F,
    typename S = std::invoke_result_t<F, T>>
  static M<S>
  map(const M<T>& mt, F f) {
    return
      Derived::flat_map(
        mt,
        [f](auto&& t) {
          return Derived::pure(f(t));
        });
  }

  template <
    typename T,
    typename F,
    typename S = std::invoke_result_t<F, T>>
  static M<S>
  map(M<T>&& mt, F f) {
    return
      Derived::flat_map(
        std::move(mt),
        [f](auto&& t) {
          return Derived::pure(f(std::move(t)));
        });
  }

  // TODO: is this correct? There's a bias toward the first argument in
  // maintaining effects in the result.
  template <typename A, typename B>
  static M<std::tuple<A, B>>
  product(const M<A>& ma, const M<B>& mb) {
    return
      Derived::flat_map(
        ma,
        [mb](auto&& a) {
          return
            Derived::flat_map(
              mb,
              [a](auto&& b) {
                return std::make_tuple(a, b);
              });
        });
  }
};

template <template <typename> typename M>
struct Monad
  : public MonadBase<Monad<M>, M> {

  template <typename T>
  struct value {
    using type = void;
  };

  template <typename T>
  using value_t = typename value<T>::type;

  template <typename T>
  static M<T>
  pure(const T& t);

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static M<S>
  flat_map(const M<T>& t, F f);

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static M<S>
  flat_map(M<T>&& t, F f);

  template <
    typename A,
    typename F,
    typename B =
      std::variant_alternative_t<1, value_t<std::invoke_result_t<F, A>>>>
  static M<B>
  tail_rec_m(A&& a, F f);
};

template <>
struct Monad<rval_t>
  : public MonadBase<Monad<rval_t>, rval_t> {

  template <typename T>
  using M = rval_t<T>;

  template <typename T>
  struct value {
    using type = void;
  };

  template <typename T>
  struct value<rval_t<T>> {
    using type = T;
  };

  template <typename T>
  using value_t = typename value<T>::type;

  template <typename T>
  static rval_t<T> pure(T&& t) {
    return rval<T>(std::forward<T>(t));
  }

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static rval_t<S>
  flat_map(const rval_t<T>& rv, F f) {
    return RvalMM::flatmap(rv, f);
  }

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static rval_t<S>
  flat_map(rval_t<T>&& rv, F f) {
    return RvalMM::flatmap(std::move(rv), f);
  }

  template <
    typename A,
    typename F,
    typename B =
      std::variant_alternative_t<1, value_t<std::invoke_result_t<F, A>>>>
  static rval_t<B>
  tail_rec_m(A&& a, F f) {
    auto mab = f(std::forward<A>(a));
    bool is_a = true;
    // be careful here to avoid anything but move construction/assignment of
    // values in mab
    while (is_value(mab) && is_a) {
      auto ab = get_value(std::move(mab));
      is_a = std::holds_alternative<A>(ab);
      if (is_a)
        mab = f(std::get<A>(std::move(ab)));
      else
        mab = std::move(ab);
    }
    if (is_value(mab))
      return rval<B>(std::get<B>(get_value(std::move(mab))));
    else
      return rval<B>(get_error(mab));
  }
};

template <template <typename> typename F>
struct functor {
  using type = void;
};

template <template <typename> typename F>
concept HasFunctor = requires {
  typename functor<F>;
};

template <template <typename> typename F>
struct applicative
  : public functor<F> {
  using type = void;
};

template <template <typename> typename F>
concept HasApplicative = requires {
  typename applicative<F>;
};

template <template <typename> typename M>
struct monad
  : public applicative<M> {
  using type = void;
};

template <template <typename> typename M>
concept HasMonad = requires {
  typename monad<M>;
};

template <>
struct monad<rval_t> {
  using type = Monad<rval_t>;
};

template <>
struct functor<rval_t> {
  using type = Monad<rval_t>;
};

template <
  template <typename> typename M,
  typename A,
  typename B>
struct Kleisli;

template <
  template <typename> typename M,
  typename A,
  typename B,
  typename F>
struct KleisliF;

template <
  template <typename> typename M,
  typename A,
  typename B>
struct Kleisli {

  template <typename F>
  static KleisliF<M, A, B, F>
  wrap(F&& f) {
    return KleisliF<M, A, B, F>(std::forward<F>(f));
  }
};

template <
  template <typename> typename M,
  typename A,
  typename B,
  typename F>
struct KleisliF {

  F m_f;

  KleisliF(const F& f)
    : m_f(f) {}

  KleisliF(F&& f)
    : m_f(std::move(f)) {}

  M<B>
  run(const A& a) {
    return m_f(a);
  }

  M<B>
  run(A&& a) {
    return m_f(std::move(a));
  }

  template <typename G> requires HasMonad<M>
  auto
  and_then(G&& g) const & {
    using C =
      typename monad<M>::type::template value_t<std::invoke_result_t<G, B>>;

    return
      Kleisli<M, A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return monad<M>::type::flat_map(f(std::forward<AA>(a)), g);
        });

  }

  template <typename G> requires HasFunctor<M>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      Kleisli<M, A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return functor<M>::type::map(f(std::forward<AA>(a)), g);
        });
  }
};

template <template <typename> typename M, typename B, typename F>
struct KleisliF<M, void, B, F> {

  F m_f;

  KleisliF(const F& f)
    : m_f(f) {}

  KleisliF(F&& f)
    : m_f(std::move(f)) {}

  M<B>
  run() {
    return m_f();
  }

  template <typename G> requires HasMonad<M>
  auto
  and_then(G&& g) const & {
    using C =
      typename monad<M>::type::template value_t<std::invoke_result_t<G, B>>;

    return
      Kleisli<M, void, C>::wrap(
        [g, f=m_f]() {
          return monad<M>::type::flat_map(f(), g);
        });
  }

  template <typename G> requires HasFunctor<M>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      Kleisli<M, void, C>::wrap(
        [g, f=m_f]() {
          return functor<M>::type::map(f(), g);
        });
  }
};

// The following classes may be of use in a functional programming style of
// error handling; otherwise, they can be safely ignored.

template <typename A, typename B, typename F>
struct RvalMF;

/** functions with domain A and range rval_t<B> */
template <typename A, typename B>
struct RvalM
  : public Kleisli<rval_t, A, B> {

  template <typename F>
  static RvalMF<A, B, F>
  wrap(F&& f) {
    return RvalMF<A, B, F>(std::forward<F>(f));
  }
};

template <typename A, typename B, typename F>
struct RvalMF
  : public KleisliF<rval_t, A, B, F> {

  using KleisliF<rval_t, A, B, F>::m_f;

  /** constructor */
  RvalMF(const F& f)
    : KleisliF<rval_t, A, B, F>(f) {}

  /** constructor */
  RvalMF(F&& f)
    : KleisliF<rval_t, A, B, F>(std::move(f)) {}

  rval_t<B>
  run(const A& a) {
    return m_f(a);
  }

  rval_t<B>
  run(A&& a) {
    return m_f(std::move(a));
  }

  template <typename G>
  auto
  and_then(G&& g) const & {
    using C =
      typename Monad<rval_t>::template value_t<std::invoke_result_t<G, B>>;

    return
      RvalM<A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return Monad<rval_t>::flat_map(f(std::forward<AA>(a)), g);
        });
  }

  template <typename G>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      RvalM<A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return Monad<rval_t>::map(f(std::forward<AA>(a)), g);
        });
  }

  template <typename G>
  auto
  and_then_loop(unsigned n, G&& g) const & {
    //using C = typename Monad<rval_t>::value<std::invoke_result_t<G, B>>::type;
    return
      and_then(
        [g, n]<typename BB>(BB&& b) {
          using ibb = std::variant<std::tuple<unsigned, B>, B>;
          return
            Monad<rval_t>::tail_rec_m(
              std::make_tuple(n, std::forward<BB>(b)),
              [g, n](auto&& i_b) -> rval_t<ibb> {
                auto& [i, b] = i_b;
                if (i == 0)
                  return ibb(std::move(b));
                else
                  return
                    Monad<rval_t>::map(
                      g(n - i, std::move(b)),
                      [i1=i - 1](auto&& b) -> ibb {
                        return std::make_tuple(i1, std::move(b));
                      });
              });
        });
  }

  template <typename G>
  auto
  and_then_repeat(unsigned n, G&& g) const & {
    return
      and_then_loop(n, [g](unsigned, auto&& b) { return g(std::move(b)); });
  }
};

/** specialization of RvalMF<A> for A = void */
template <typename B, typename F>
struct RvalMF<void, B, F>
  : public KleisliF<rval_t, void, B, F> {

  using KleisliF<rval_t, void, B, F>::m_f;

  /** constructor */
  RvalMF(const F& f)
    : KleisliF<rval_t, void, B, F>(f) {}

  /** constructor */
  RvalMF(F&& f)
    : KleisliF<rval_t, void, B, F>(std::move(f)) {}

  rval_t<B>
  run() {
    return m_f();
  }

  template <typename G>
  auto
  and_then(G&& g) const & {
    using C =
      typename Monad<rval_t>::template value_t<std::invoke_result_t<G, B>>;

    return
      RvalM<void, C>::wrap(
        [g, f=m_f]() {
          return Monad<rval_t>::flat_map(f(), g);
        });
  }

  template <typename G>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      RvalM<void, C>::wrap(
        [g, f=m_f]() {
          return Monad<rval_t>::map(f(), g);
        });
  }

  template <typename G>
  auto
  and_then_loop(unsigned n, G&& g) const & {
    //using C = typename Monad<rval_t>::value<std::invoke_result_t<G, B>>::type;
    return
      and_then(
        [g, n]<typename BB>(BB&& b) {
          using ibb = std::variant<std::tuple<unsigned, B>, B>;
          return
            Monad<rval_t>::tail_rec_m(
              std::make_tuple(n, std::forward<BB>(b)),
              [g, n](auto&& i_b) -> rval_t<ibb> {
                auto& [i, b] = i_b;
                if (i == 0)
                  return ibb(std::move(b));
                else
                  return
                    Monad<rval_t>::map(
                      g(n - i, std::move(b)),
                      [i1=i - 1](auto&& b) -> ibb {
                        return std::make_tuple(i1, std::move(b));
                      });
              });
        });
  }

  template <typename G>
  auto
  and_then_repeat(unsigned n, G&& g) const & {
    return
      and_then_loop(n, [g](unsigned, auto&& b) { return g(std::move(b)); });
  }
};

#endif // HPG_API >= 17

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
